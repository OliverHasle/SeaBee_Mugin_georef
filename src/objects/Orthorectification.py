import os
import numpy as np
import tools.constants as c

from tqdm             import tqdm
from pyproj           import Transformer
from osgeo            import gdal, osr
#from rasterio.control import GroundControlPoint as gcp

class Orthorectification:
    def __init__(self, config, parameter, dem, geoPose):
        self.config                = config
        self.parameter             = parameter
        self.geoPose               = geoPose
        self.dem                   = dem
        self.image_list            = []
        self.uav_azimuth           = []
        self.image_coordinates_utm = None
        self.uav_azimuth           = None
        self.epsg_code             = int(self.config['SETTINGS']['output_epsg'])
        self.downscale_factor      = int(self.config['SETTINGS']['downscale_factor_imgs']) if int(self.config['SETTINGS']['downscale_factor_imgs']) >= 1 else None
        self.tuning_values         = [np.float32(self.config['TUNING']['delta_north']), 
                                      np.float32(self.config['TUNING']['delta_east']), 
                                      np.float32(self.config['TUNING']['rotation']),
                                      np.float32(self.config['TUNING']['delta_flight_dir']),
                                      np.float32(self.config['TUNING']['delta_perpend_flight_dir'])]

        # Check if the image input path exists
        if os.path.exists(self.config['MISSION']['inputfolder']) and os.path.isdir(self.config['MISSION']['inputfolder']):
            # loop through all files
            files = os.listdir(self.config['MISSION']['inputfolder'])
            for file in files:
                # Check if file ends with *.jpg or *.png
                if file.lower().endswith(('.jpg', '.png')):
                    self.image_list.append(file)
            # make list unmutable (tuple), so that it cannot be changed
            self.image_list = tuple(self.image_list)

        if not self.geoPose == None:
            self.calculate_uav_azimuth()

    def load_geopose(self, geoPose):
        self.geoPose = geoPose

    def load_dem(self, dem):
        self.dem = dem

    def calculate_uav_azimuth(self):
        """
        Calculate the UAV azimuth angle based on the camera orientation
          -> azimuth => Angle between UAV heading and the North direction
        """
        if self.geoPose is None:
            print("No GeoPose object loaded! \n     -> Load GeoPose object first! \n     -> Use <Orthorectification>.load_geopose(<GeoPose>) \n     -> <Orthorectification>.synchronize_image_geopose() \n     -> <Orthorectification>.calculate_uav_azimuth()")
            return

        self.uav_azimuth = np.zeros(len(self.geoPose.image_name))
        for i in range(len(self.geoPose.yaw)):
            # Get the yaw angle from the GeoPose object
            self.uav_azimuth[i] = self.geoPose.yaw[i]

    def georectify(self, idx, imageName, verbose=False):
        """"
            Project the image onto the DEM-Mesh
        """
        image_path = os.path.join(self.config['MISSION']['inputfolder'], imageName)
        if not os.path.exists(image_path):
            #raise FileNotFoundError(f"Input image not found: {image_path}")
            print(f"Input image {imageName} not found")
            return

        # Open the image with GDAL
        image_data = gdal.Open(image_path)
        if image_data is None:
            raise FileNotFoundError(f"Could not open image {image_path}")
        if self.downscale_factor is not None:
            # Downscale the image 
            image_data = self._downscale_image(image_data, self.downscale_factor)

        # Output path / name for the georeferenced image
        output_name = imageName.split('.')[0] + '.tif'
        output_path = os.path.join(self.config['MISSION']['outputfolder'], output_name)

        # Read raster data as numeric array from GDAL Dataset into a numpy array [x-offset, y-offset, x-width (columns), y-width (rows)]
        I            = image_data.ReadAsArray(0, 0, image_data.RasterXSize, image_data.RasterYSize)
        mirror_image = self.config['SETTINGS']['mirror_images'].lower()
        if mirror_image == 'horizontal' or mirror_image == 'both':
            I = self._mirror_image(I, 'x')
        elif mirror_image == 'vertical' or mirror_image == 'both':
            I = self._mirror_image(I, 'y')

        # Fetch a driver based on the short name (GTiff)
        outdataset = gdal.GetDriverByName('GTiff')

        # Create a new dataset (this creates a new file)
        outdataset  = outdataset.Create(output_path,
                                        xsize=image_data.RasterXSize,
                                        ysize=image_data.RasterYSize,
                                        bands=image_data.RasterCount,
                                        eType=gdal.GDT_Byte)

        # Image coordinates in UTM coordinates [top_left, top_right, bottom_left, bottom_right, center]
        img_azim_left, img_azim_right  = self._calculate_image_rotation(self.image_coordinates_utm[idx, :, :])
        heading                        = np.deg2rad(self.uav_azimuth[idx])
        flight_dir_azim                = self._calc_uav_azimuth_gnd_trk(idx)
        geotransform                   = self._calculate_geotransform(self.image_coordinates_utm[idx, :, :],
                                                                      flight_dir_azim,
                                                                      [img_azim_left, img_azim_right], 
                                                                      image_data.RasterXSize,
                                                                      image_data.RasterYSize,
                                                                      tuning_values=self.tuning_values)
        # Rotation of the image in Radians (0° = y-axis facing North)

        outdataset.SetGeoTransform(geotransform)

        # Create new Spatial Reference System (SRS) object
        output_SRS = osr.SpatialReference()
        # Set the output SRS to the WGS84 geocentric coordinate system
        # -> This defines the coordinate system of the output image
        output_SRS.ImportFromEPSG(self.epsg_code)
        # Set the projection of the output image
        outdataset.SetProjection(output_SRS.ExportToWkt())

        # Write the bands
        if image_data.RasterCount == 1:
            outdataset.GetRasterBand(1).WriteArray(I)
        else:
            for band in range(image_data.RasterCount):
                outdataset.GetRasterBand(band + 1).WriteArray(I[band,:,:])

        # Flush image and clean up
        outdataset.FlushCache()
        outdataset = None

        if verbose == True:
            print(f"Image {imageName} has been orthorectified and saved as {output_name} in the output folder!")

    def georectify_all(self, verbose=False):
        """
        Orthorectify all images in the image list
        INPYTS:
        """
        # Check in the config.ini file if the user wants to overwrite the output files
        overwrite = True if self.config['SETTINGS']['overwrite_output'].lower() == 'true' else False

        # Make sure the output folder exists
        if os.path.exists(os.path.join(self.config['MISSION']['outputfolder'])):
            # Check if there are files in the output folder
            files = os.listdir(self.config['MISSION']['outputfolder'])
            if len(files) > 0:
                print("Output folder is not empty!")
                # Ask if the user wants to overwrite the files
                if overwrite == False:
                    user_input = input("Do you want to continue? [y/n]: ")
                    if user_input.lower() == 'n':
                        overwrite == False
                        print("Orthorectification aborted!")
                        return
                    elif user_input.lower() == 'y':
                        overwrite == True
                    else:
                        print("Invalid input! Orthorectification aborted!")
                        return

                if overwrite == True:
                    print("  --  Files will be overwritten!  --")   
                    # Remove all files in the output folder
                    for file in files:
                        os.remove(os.path.join(self.config['MISSION']['outputfolder'], file))
                else:
                    print("Orthorectification aborted!")
                    return

        else:
            print("Output folder does not exist!")
            # Create the output folder
            os.makedirs(self.config['MISSION']['outputfolder'])
            print(f"Output folder {self.config['MISSION']['outputfolder']} has been created!")

        # Convert the ECEF coordinates to UTM coordinates
        self.image_coordinates_utm = self._convert_ECEF_to_UTM(self.geoPose.p_eg_e, self.epsg_code)

        # Orthorectify all images
        print("Orthorectifying images...")
        for idx in tqdm(range(len(self.geoPose.image_name))):
            image_name = self.geoPose.image_name[idx]
            self.georectify(idx, image_name, verbose=verbose)
        print("All images have been orthorectified!")

    def _calc_uav_azimuth_gnd_trk(self, idx, north_vector=np.array([0, 1])):
        """ 
        Calculate the UAV azimuth in radians in the specified epsg coorinate system
        -> calculation based on the fligh direction, NOT the measured UAV heading
        -> Clockwise is positive
        """
        if self.geoPose is None:
            print("No GeoPose object loaded! \n     -> Load GeoPose object first! \n     -> Use <Orthorectification>.load_geopose(<GeoPose>)")
            return
        # make sure 
        if idx < len(self.geoPose.p_eg_e) - 1:
            p_utm_k0 = self.image_coordinates_utm[idx, 4, 0:2]
            p_utm_k1 = self.image_coordinates_utm[idx + 1, 4, 0:2]
        else:
            p_utm_k0 = self.image_coordinates_utm[idx - 1, 4, 0:2]
            p_utm_k1 = self.image_coordinates_utm[idx, 4, 0:2]


        # Calculate the flight direction vector (Northing, Easting)
        flight_direction = p_utm_k1 - p_utm_k0
        flight_direction = flight_direction / np.linalg.norm(flight_direction)

        # Calculate the azimuth angle / Coordinate convention is [Easting, Northing]
        azimuth = np.arccos(np.dot(flight_direction, north_vector))

        # Determine the sign of the angle
        if np.cross(flight_direction, north_vector) < 0:
            azimuth = -azimuth
        return azimuth

    @staticmethod
    def _convert_ECEF_to_UTM(ecef_geocent, epgs_code): # TODO: make this function non-static
        """
        Convert ECEF coordinates to UTM coordinates
        """

        epgs_string_to   = f"epsg:{epgs_code}"
        epgs_string_from = f"epsg:{c.EPSG_geocent}"
        wgs84_to_utm     = Transformer.from_crs(epgs_string_from, epgs_string_to, always_xy=True)

        utm_coord = np.zeros_like(ecef_geocent)

        if len(ecef_geocent.shape) == 1:
            # One dimensional array
            easting, northing, alt = wgs84_to_utm.transform(ecef_geocent[0], ecef_geocent[1], ecef_geocent[2])
            utm_coord              = [easting, northing, alt]
        if len(ecef_geocent.shape) == 2:
            # Two dimensional array
            for meas_i in range(ecef_geocent.shape[0]):
                easting, northing, alt = wgs84_to_utm.transform(ecef_geocent[meas_i, 0], ecef_geocent[meas_i, 1], ecef_geocent[meas_i, 2])
                utm_coord[meas_i, :]   = [easting, northing, alt]
        elif len(ecef_geocent.shape) == 3:
            # Three dimensional array
            for meas_i in range(ecef_geocent.shape[0]):
                for vec_i in range(ecef_geocent.shape[1]):
                    easting, northing, alt      = wgs84_to_utm.transform(ecef_geocent[meas_i, vec_i, 0], ecef_geocent[meas_i, vec_i, 1], ecef_geocent[meas_i, vec_i, 2])
                    utm_coord[meas_i, vec_i, :] = [easting, northing, alt]
        else:
            print("ERROR: ECEF coordinates have wrong shape!")
            return None

        return utm_coord

    @staticmethod
    def _downscale_image(image_data, downscale_factor):
        """
        This function downscales an image by a certain factor
        """
        # Get the image size
        width  = image_data.RasterXSize
        height = image_data.RasterYSize

        # Calculate the new image size
        new_width  = width // downscale_factor
        new_height = height // downscale_factor

        gt = list(image_data.GetGeoTransform())
        gt[1] *= downscale_factor  # Adjust pixel width
        gt[5] *= downscale_factor  # Adjust pixel height

        # Create output dataset
        driver = gdal.GetDriverByName('MEM')
        output_dataset = driver.Create('', 
                                     new_width, 
                                     new_height, 
                                     image_data.RasterCount, 
                                     image_data.GetRasterBand(1).DataType)

        # Perform the resampling for each band
        for band in range(1, image_data.RasterCount + 1):
            input_band  = image_data.GetRasterBand(band)
            output_band = output_dataset.GetRasterBand(band)

            gdal.RegenerateOverview(input_band,
                                  output_band,
                                  'AVERAGE')

        return output_dataset
    
    @staticmethod
    def _calculate_geotransform(image_corners, uav_azimuth, img_azim, width_pixel, height_pixel, tuning_values=None):
        """
        Calculate geotransform from corner coordinates and heading

        Parameters:
        image_corners: list of coordinates in order [top_left, top_right, bottom_left, bottom_right, center]
        azimuth:       heading angle in degrees (0 = north, clockwise positive)
        width_pixel:   image width in pixels
        height_pixel:  image height in pixels
        """
        # Shift the images in flight direction
        flight_direction        = np.array([np.sin(uav_azimuth), np.cos(uav_azimuth), 0])
        shift_along_flight_dir  = tuning_values[3]
        shift_across_flight_dir = tuning_values[4]
        shift_fl_dir            = shift_along_flight_dir * flight_direction + shift_across_flight_dir * np.array([-flight_direction[1], flight_direction[0], 0])

        top_left     = image_corners[0,:]  + np.array([tuning_values[0], tuning_values[1], 0]) + shift_fl_dir          # Add tuning values
        top_right    = image_corners[1,:]  + np.array([tuning_values[0], tuning_values[1], 0]) + shift_fl_dir          # Add tuning values
        bottom_left  = image_corners[2,:]  + np.array([tuning_values[0], tuning_values[1], 0]) + shift_fl_dir          # Add tuning values
        bottom_right = image_corners[3,:]  + np.array([tuning_values[0], tuning_values[1], 0]) + shift_fl_dir          # Add tuning values # Unused
        center       = image_corners[4,:]  + np.array([tuning_values[0], tuning_values[1], 0]) + shift_fl_dir          # Add tuning values # Unused

        avg_image_rotation_rad = 0.5*(img_azim[0] + img_azim[1]) + np.deg2rad(tuning_values[2])                        # Rotation of the image in Radians (0° = y-axis facing North)

        # Calculate image dimensions from corners
        # With of the image in meters
        width_meter  = np.sqrt((top_right[0]   - top_left[0])**2 + 
                               (top_right[1]   - top_left[1])**2)
        # Height of the image in meters
        height_meter = np.sqrt((bottom_left[0] - top_left[0])**2 + 
                               (bottom_left[1] - top_left[1])**2)

        # Calculate pixel size in meters along the x and y axis of the image
        pixel_size_x  = width_meter / width_pixel
        pixel_size_y  = height_meter / height_pixel

        # Calculate rotation terms        # Top left coordinates # TODO: ADD TUNING
        g0 = top_left[0]                                         # GT[0] => Pixel position of the "top left pixel" in UTM coordinates (X, Easting)
        g1 = pixel_size_x    * np.cos(avg_image_rotation_rad)    # GT[1] => Represents how much X (easting) changes when you move one pixel to the right (P = P + 1)
        g2 = -pixel_size_x   * np.sin(avg_image_rotation_rad)    # GT[2] => Represents how much X (easting) changes when you move one row down (L = L + 1)            (0 for north-up and no rotation)
        g3 = top_left[1]                                         # GT[3] => Pixel position of the "top left pixel" in UTM coordinates (Y, Northing)
        g4 = -pixel_size_y   * np.sin(avg_image_rotation_rad)    # GT[4] => Represents how much Y (northing) changes when you move one pixel to the right (P = P + 1) (0 for north-up and no rotation)
        g5 = -pixel_size_y * np.cos(avg_image_rotation_rad)      # GT[5] => Represents how much Y (northing) changes when you move one row down (L = L + 1)

        return (g0, g1, g2, g3, g4, g5)

    @staticmethod
    def _calculate_image_rotation(coordinates, northing_ref_vect=np.array([0, 1]), debug=False):
        """
        Calculate the rotation of the image based on its corner coordinates

        INPUTS:
        top_left, top_right, bottom_left, bottom_right, center: UTM coordinates of image corners

        RETURNS:
        Rotation angle in degrees
        """
        # Protection against wrong input shape (5x3)
        if coordinates.shape[0] != 5 or coordinates.shape[1] != 3:
            raise ValueError("Wrong input shape for image coordinates!")

        # Calculate the left and right azimuth angles
        left_img_edge          = coordinates[0,:] - coordinates[2,:]                               # top_left - bottom_left -> "left edge"
        left_img_edge          = left_img_edge[0:2]                                                # Ignore the altitude]
        left_edge_unit_vector  = left_img_edge / np.linalg.norm(left_img_edge)                     # Normalize the vector
        
        right_img_edge         = coordinates[1,:] - coordinates[3,:]                               # top_right - bottom_right -> "right edge"
        right_img_edge         = right_img_edge[0:2]                                               # Ignore the altitude
        right_edge_unit_vector = right_img_edge / np.linalg.norm(right_img_edge)                   # Normalize the vector

        northing_ref_vect_unit = northing_ref_vect / np.linalg.norm(northing_ref_vect)             # Normalize the reference vector (Northing)

        left_cos_angle      = np.dot(left_edge_unit_vector, northing_ref_vect_unit)
        left_azim           = np.arccos(left_cos_angle)

        right_cos_angle     = np.dot(right_edge_unit_vector, northing_ref_vect_unit)
        right_azim          = np.arccos(right_cos_angle)

        # Determine the sign of the angle
#        if left_edge_unit_vector[0] < 0:
#            left_azim = -left_azim
        if  np.cross(left_edge_unit_vector, northing_ref_vect_unit) < 0:
            left_azim = -left_azim
        if np.cross(right_edge_unit_vector, northing_ref_vect_unit) < 0:
            right_azim = -right_azim

        if debug == True:
            print(f"Rotation angle: {left_azim * 180 / np.pi}°")
            import matplotlib.pyplot as plt
            plt.plot(coordinates[:,0], coordinates[:,1], 'ro')
            # Plot the vectors with the origin int the middle and arows
            plt.quiver(coordinates[4,0], coordinates[4,1], left_edge_unit_vector[0], left_edge_unit_vector[1], angles='xy', scale_units='xy', scale=1, color='blue')
            plt.quiver(coordinates[4,0], coordinates[4,1], northing_ref_vect_unit[0], northing_ref_vect_unit[1], angles='xy', scale_units='xy', scale=1, color='green')       # Number the corners
            for i in range(5):
                plt.text(coordinates[i,0], coordinates[i,1], str(i), fontsize=12, color='black')
            plt.show()

        return left_azim, right_azim

    @staticmethod
    def _mirror_image(I, axis):
        """
        Mirror the image along the specified axis
        """
        if axis == 'y':
            if I.ndim == 3:
                return I[:, ::-1, :]
            if I.ndim == 2:
                return I[::-1, :]
        elif axis == 'x':
            if I.ndim  == 3:
                return I[:, :, ::-1]
            if I.ndim == 2:
                return I[:, ::-1]
        else:
            # No mirroring
            return I
