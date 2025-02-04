import os
import numpy as np
import tools.constants as c

from tqdm             import tqdm
from pyproj           import Transformer
from osgeo            import gdal, osr
from rasterio.control import GroundControlPoint as gcp

class Orthorectification:
    def __init__(self, config, parameter, dem=None, geoPose=None):
        self.config                = config
        self.parameter             = parameter
        self.geoPose               = geoPose
        self.dem                   = dem
        self.image_list            = []
        self.idx_list_geopose      = []
        self.uav_azimuth           = []
        self.image_coordinates_utm = None
        self.uav_azimuth           = None
        self.epsg_code             = int(self.config['SETTINGS']['output_epsg'])
        self.downscale_factor      = int(self.config['SETTINGS']['downscale_factor_imgs']) if int(self.config['SETTINGS']['downscale_factor_imgs']) >= 1 else None

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
            self.synchronize_image_geopose()
            self.calculate_uav_azimuth()

    def load_geopose(self, geoPose):
        self.geoPose = geoPose

    def load_dem(self, dem):
        self.dem = dem

    def synchronize_image_geopose(self):
        if self.geoPose is None:
            print("No GeoPose object loaded! \n     -> Load GeoPose object first! \n     -> Use <Orthorectification>.load_geopose(<GeoPose>)")
            return
        if self.image_list is None:
            print("Image list is empty!!")
            return

        for image in self.image_list:
            # find the index of the image name in the geopose object
            if image not in self.geoPose.image_name:
                print(f"Image {image} not found in GeoPose object!")
                continue
            # Get the index of the image in the geopose object
            idx = np.where(self.geoPose.image_name == image)
            if len(idx) > 1:
                print(f"WARNING: Multiple images with the same name found in GeoPose object, => Check gpslog.txt!")
                continue
            self.idx_list_geopose.append(int(idx[0]))

    def calculate_uav_azimuth(self):
        """
        Calculate the UAV azimuth angle based on the camera orientation
          -> azimuth => Angle between UAV heading and the North direction
        """
        if self.geoPose is None:
            print("No GeoPose object loaded! \n     -> Load GeoPose object first! \n     -> Use <Orthorectification>.load_geopose(<GeoPose>) \n     -> <Orthorectification>.synchronize_image_geopose() \n     -> <Orthorectification>.calculate_uav_azimuth()")
            return
        if self.idx_list_geopose is None:
            print("No images synchronized with GeoPose object! \n     -> Synchronize images with GeoPose object first! \n     -> Use <Orthorectification>.synchronize_image_geopose() \n     -> <Orthorectification>.calculate_uav_azimuth()")
            return

        self.uav_azimuth = np.zeros(len(self.idx_list_geopose))
        for i in range(len(self.idx_list_geopose)):
            # Get the yaw angle from the GeoPose object
            self.uav_azimuth[i] = self.geoPose.yaw[i]

    def georectify(self, idx, verbose=False):
        """"
            Project the image onto the DEM-Mesh
        """
        image_name = self.image_list[idx]
        image_path = os.path.join(self.config['MISSION']['inputfolder'], image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # Open the image with GDAL
        image_data = gdal.Open(image_path)
        if image_data is None:
            raise FileNotFoundError(f"Could not open image {image_path}")
        if self.downscale_factor is not None:
            # Downscale the image 
            image_data = self._downscale_image(image_data, self.downscale_factor)

        # Output path / name for the georeferenced image
        output_name = image_name.split('.')[0] + '.tif'
        output_path = os.path.join(self.config['MISSION']['outputfolder'], output_name)

        # Read raster data as numeric array from GDAL Dataset into a numpy array [x-offset, y-offset, x-width (columns), y-width (rows)]
        I            = image_data.ReadAsArray(0, 0, image_data.RasterXSize, image_data.RasterYSize)
        mirror_image = self.config['SETTINGS']['mirror_images'].lower()
        if mirror_image == 'horizontal':
            I = self._mirror_image(I, 'x')
        elif mirror_image == 'vertical':
            I = self._mirror_image(I, 'y')

        # Fetch a driver based on the short name (GTiff)
        outdataset = gdal.GetDriverByName('GTiff')

        # Create a new dataset (this creates a new file)
        outdataset  = outdataset.Create(output_path,
                                        xsize=image_data.RasterXSize,
                                        ysize=image_data.RasterYSize,
                                        bands=image_data.RasterCount,
                                        eType=gdal.GDT_Float32)

        top_left    = self.image_coordinates_utm[idx, 0, :]
        top_right   = self.image_coordinates_utm[idx, 1, :]
        bottom_left = self.image_coordinates_utm[idx, 2, :]

        # Calculate pixel width and height
        pixel_width  = np.linalg.norm(top_right[0:2]   - top_left[0:2]) / image_data.RasterXSize
        pixel_height = np.linalg.norm(bottom_left[0:2] - top_left[0:2]) / image_data.RasterYSize


        azimuth_rad = np.deg2rad(self.uav_azimuth[idx])
        rot_matrix = np.array([[np.cos(azimuth_rad), -np.sin(azimuth_rad)], 
                              [np.sin(azimuth_rad), np.cos(azimuth_rad)]])
        
        # Calculate the new top_left coordinate
#        rot_top_left = np.array([top_left[0], top_left[1]])
        rot_top_left = rot_matrix @ np.array([top_left[0], top_left[1]])
        #rot_top_right = np.array([top_right[0], top_right[1]])
        rot_top_right = rot_matrix @ np.array([top_right[0], top_right[1]])
        #rot_bottom_left = np.array([bottom_left[0], bottom_left[1]])
        rot_bottom_left = rot_matrix @ np.array([bottom_left[0], bottom_left[1]])

#        geotransform = [
#            top_left[0],                                 # top left x coordinate
#            pixel_width * np.cos(azimuth_rad),    # west-east pixel resolution [m] adjusted for rotation
#            -pixel_width * np.sin(azimuth_rad),   # rotation/skew 
#            top_left[1],                                 # top left y coordinate
#            pixel_height * np.sin(azimuth_rad),   # rotation 
#            -pixel_height * np.cos(azimuth_rad)   # north-south pixel resolution [m] adjusted for rotation
#        ]

        geotransform = [
            top_left[0],                   # top left x coordinate
            pixel_width,                   # west-east pixel resolution [m]
            0,                             # rotation/skew [degrees]
            top_left[1],                   # top left y coordinate
            0,                             # rotation [degrees]
            -pixel_height                  # north-south pixel resolution [m] (negative value for north-up image)
        ]

#        rotation_angle = self._calculate_image_rotation(self.image_coordinates_utm[idx, :, :])
#        rotation_angle_rad = np.deg2rad(rotation_angle)
#        geotransform = [
#            top_left[0],                                 # top left x coordinate
#            pixel_width * np.cos(rotation_angle_rad),    # west-east pixel resolution [m] adjusted for rotation
#            -pixel_width * np.sin(rotation_angle_rad),   # rotation/skew 
#            top_left[1],                                 # top left y coordinate
#            pixel_height * np.sin(rotation_angle_rad),   # rotation 
#            -pixel_height * np.cos(rotation_angle_rad)   # north-south pixel resolution [m] adjusted for rotation
#        ]

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
            print(f"Image {image_name} has been orthorectified and saved as {output_name} in the output folder!")

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
        for idx in tqdm(self.idx_list_geopose):
            self.georectify(idx, verbose=verbose)
        print("All images have been orthorectified!")

    @staticmethod
    def _calculate_image_rotation(coordinates, ref_vector=np.array([0, 1])):
        """
        Calculate the rotation of the image based on its corner coordinates
        
        INPUTS:
        top_left, top_right, bottom_left, bottom_right, center: UTM coordinates of image corners
        
        RETURNS:
        Rotation angle in degrees
        """
        # TDOO Protection against wrong input shape
        # TODO Protection against "skewed" image in altitude direction


        # Calculate the top edge vector
        top_edge_vector      = coordinates[1,:] - coordinates[0,:]                #top_right - top_left
        top_edge_vector      = top_edge_vector[0:2]                               # Ignore the altitude]
        top_edge_unit_vector = top_edge_vector / np.linalg.norm(top_edge_vector)
        ref_vector_unit      = ref_vector / np.linalg.norm(ref_vector)

        cos_angle      = np.dot(top_edge_unit_vector, ref_vector_unit)
        rotation_angle = np.arccos(cos_angle) * 180 / np.pi

        cross_product = np.cross(ref_vector_unit, top_edge_unit_vector)
        if cross_product < 0:
            rotation_angle = -rotation_angle

        return rotation_angle

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
            northing, easting, alt = wgs84_to_utm.transform(ecef_geocent[0], ecef_geocent[1], ecef_geocent[2])
            utm_coord              = [northing, easting, alt]
        if len(ecef_geocent.shape) == 2:
            # Two dimensional array
            for meas_i in range(ecef_geocent.shape[0]):
                northing, easting, alt = wgs84_to_utm.transform(ecef_geocent[meas_i, 0], ecef_geocent[meas_i, 1], ecef_geocent[meas_i, 2])
                utm_coord[meas_i, :]   = [northing, easting, alt]
        elif len(ecef_geocent.shape) == 3:
            # Three dimensional array
            for meas_i in range(ecef_geocent.shape[0]):
                for vec_i in range(ecef_geocent.shape[1]):
                    northing, easting, alt      = wgs84_to_utm.transform(ecef_geocent[meas_i, vec_i, 0], ecef_geocent[meas_i, vec_i, 1], ecef_geocent[meas_i, vec_i, 2])
                    utm_coord[meas_i, vec_i, :] = [northing, easting, alt]
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