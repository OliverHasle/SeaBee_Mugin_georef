import os
import cv2
import fiona
import rasterio
import numpy                as np
import tools.visualizations as vis
import json

#from rasterio.mask    import mask
from tqdm             import tqdm
from osgeo            import gdal, ogr, osr
from pathlib          import Path
#from typing           import Dict
#from scipy.signal     import correlate2d, fftconvolve
#from scipy.fft        import fft2, ifft2

#from scipy.optimize import minimize

from shapely.geometry import Polygon, MultiPolygon
#from shapely.ops import unary_union

class FeatureMatching:
    def __init__(self, config):
        self.config     = config
        self.images     = {} # Image dictionary containing all the image information
        
        # Initialize the image list, offsets and confidence
        self.image_list = []
        self.offsets    = []
        self.confidence = []
        self.mean_mag   = []
        self.std_div    = []
        self.processed_set = set()

        self._load_geotiffs()
        self._clear_orthorectification_folder()

    def _load_geotiffs(self) -> None:
        """
        Loading all geotiff images in the image directory specified in the configuration file.
        """
        file_path = Path(self.config["MISSION"]["outputfolder"])
        print("Loading images from", file_path)

        try:
            tif_files   = list(file_path.glob("*.tif"))
            if not tif_files:
                raise FileNotFoundError("No geotiff files found in the specified directory.")

            self.images = {}

            for file in tqdm(tif_files, desc="Loading Geotiffs"):
                try:
                    # Create image information dirtionary for each image
                    # Load the image using gdal and store it in the dictionary
                    src = gdal.Open(str(file))
                    self.images[str(file)] = {
                        "filepath": str(file),
                        "gdalImg":  src
                    }
                except Exception as e:
                    print(f"Error loading {file}")
                    print(e)

            print("Loaded", len(self.images), "images.")
        except FileNotFoundError as e:
            print(e)

    def _clear_orthorectification_folder(self) -> None:
        """
        Clear the orthorectification folder before saving new images.
        """
        folder_name = self.config["MISSION"]["orthorectification_folder"]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            for file in os.listdir(folder_name):
                file_path = os.path.join(folder_name, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}")
                    print(e)

    def process_image_grid(self):
        """
        Process images from NW to SE, calculating required offsets.
        """
        # Convert dictionary values to a list for sorting purposes
        self.image_list = list(self.images.values())

        # Visualize the georeferenced images (for debugging)
#        vis.plot_georeferenced_images(image_list, first_idx=0, last_idx=2, title='Georeferenced Mugin Images', cmap='terrain', figsize=(10, 8), show_overlap=True)

        # Look through all images 
        print("---------------------------------------------------")
        for i, img in enumerate(tqdm(self.image_list, desc="Shifting Images using optical flow", unit="image")):
            if i == 0:
                # First image has no shift
                prev_shifts = self._process_image_with_no_shift(img, i)
                continue

            overlap_shape, overlap_sr = self._get_overlap_shape(self.image_list[i-1]["filepath"],    # Base image (n-1)
                                                                self.image_list[i]["filepath"],      # Current image (n)
                                                                buffer_size=5)                  # Buffer around the overlap in meters

            if overlap_shape is None:
                # If there is no overlap continue to the next image and save the image without any changes
                prev_shifts = self._process_image_with_no_shift(img, i)
                continue

            overlap_n_1 = self._cut_image(self.image_list[i-1], overlap_shape, overlap_sr)
            overlap_n   = self._cut_image(self.image_list[i],   overlap_shape, overlap_sr)

            # Get the image data of the image i
            #x_shift, y_shift, stat = self._shift_opticalFlow(overlap_n_1,
            #                                                 overlap_n)
            x_shift, y_shift, stat = self._shift_opticalFlow_multiscale(overlap_n_1,
                                                                        overlap_n)

            # Store the shift for the image
            self.offsets.append(np.array([x_shift, y_shift]))
            if stat is None:
                self.confidence.append(0)
                self.mean_mag.append(0)
                self.std_div.append(0)
            else:
                self.confidence.append(stat[0])
                self.mean_mag.append(stat[1])
                self.std_div.append(stat[2])

            # Calculate the total shift
            total_shift = prev_shifts + np.array([x_shift, y_shift])
            prev_shifts = total_shift

            # Apply the shift to the image
            img_name = self.image_list[i]["filepath"]
            self._apply_shift(self.images[img_name], total_shift[0], total_shift[1])
            self.processed_set.add(i)

        print("Processing done.")
        print("---------------------------------------------------")
        #Show all the offsets in a graph
        vis.plot_offsets(self.offsets, self.confidence, self.mean_mag, self.std_div, title="Shifts between images", figsize=(10, 8))

    def _process_image_with_no_shift(self, img, index):
        # Save the image without any changes
        img_name = os.path.basename(img["filepath"])
        img_name = img_name.replace(".tif", "_ortho.tif")
        self._save_image(img_name, gdal_img=img["gdalImg"])

        # Shift for this image is zero
        self.processed_set.add(index)

        self.offsets.append(np.array([0, 0]))
        self.confidence.append(0)
        self.mean_mag.append(0)
        self.std_div.append(0)
    
        # Return zero shift for tracking
        return np.array([0, 0])

    def _get_overlap_shape(self, img_name1, img_name2, buffer_size=0, save_output_to_file=False):
        """
        Creates a polygon representing the overlap between two images.

        Parameters:
        img_name1 (str): Name/path of the first image
        img_name2 (str): Name/path of the second image
        edge_frame_width (int): Optional buffer around the overlap in pixels

        Returns:
        overlap_poly: Shapely polygon representing the overlap region
        sr: Spatial reference system for the polygon
        """
        # Helper function: Create polygons
        def create_image_polygon(gdal_img, geotransform):
            """Helper function to create a polygon for an image"""
            width = gdal_img.RasterXSize
            height = gdal_img.RasterYSize

            # Get the corner coordinates in pixel space
            corners = [(0, 0), (width, 0), (width, height), (0, height)]

            # Transform corners to world coordinates
            world_corners = []
            for x, y in corners:
                world_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
                world_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
                world_corners.append((world_x, world_y))

            # Create a Shapely polygon
            return Polygon(world_corners)
        
        base   = self.images[img_name1]["gdalImg"]
        target = self.images[img_name2]["gdalImg"]

        # Get geotransforms
        gt_base   = base.GetGeoTransform()
        gt_target = target.GetGeoTransform()

        # Get spatial reference system from base image
        sr = osr.SpatialReference()
        sr.ImportFromWkt(base.GetProjection())

        # Create polygons for both images
        base_poly   = create_image_polygon(base, gt_base)
        target_poly = create_image_polygon(target, gt_target)

        # Calculate the intersection (overlap)
        if not base_poly.intersects(target_poly):
            #print("No overlap between images")
            return None, None

        overlap_poly = base_poly.intersection(target_poly)

        # If the result is a MultiPolygon, take the largest polygon
        if isinstance(overlap_poly, MultiPolygon):
            if len(overlap_poly.geoms) > 1:
                print(f"Warning: Multiple overlap regions found. Using the largest one.")
            largest_poly = max(overlap_poly.geoms, key=lambda p: p.area)
            overlap_poly = largest_poly

        # Check if the overlap is valid
        if overlap_poly.is_empty or overlap_poly.area <= 0:
            print("No valid overlap between images")
            return None, None

        # Store the exact overlap before adding frame
        if buffer_size > 0:
            # Add a buffer around the overlap polygon
            buffered_poly = overlap_poly.buffer(buffer_size)
            overlap_poly = buffered_poly.intersection(base_poly.union(target_poly))

        # Convert Shapely polygon to OGR geometry
        overlap_coords = list(overlap_poly.exterior.coords)

        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in overlap_coords:
            ring.AddPoint(x, y)

        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)

        # Add interior rings (holes) if any
        for interior in overlap_poly.interiors:
            interior_ring = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in interior.coords:
                interior_ring.AddPoint(x, y)
            polygon.AddGeometry(interior_ring)

        # Assign the spatial reference to the polygon
        polygon.AssignSpatialReference(sr)

        if save_output_to_file:
            folder_name    = self.config["MISSION"]["orthorectification_folder"]
            img_file_name1 = self.images[img_name1]["filepath"]
            img_file_name2 = self.images[img_name2]["filepath"]

            # Get filenames without extension from image 1 and image 2
            filename1 = os.path.splitext(os.path.basename(img_file_name1))[0]
            filename2 = os.path.splitext(os.path.basename(img_file_name2))[0]

            # Add buffer information to filename
            buffer_suffix   = f"_buffer{buffer_size}" if buffer_size > 0 else ""
            output_filename = f"{filename1}_{filename2}_overlap{buffer_suffix}.shp"
            file_path       = os.path.join(folder_name, output_filename)

            # Delete existing file if it exists
            driver = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(file_path):
                driver.DeleteDataSource(file_path)

            # Create new shapefile
            ds    = driver.CreateDataSource(file_path)
            layer = ds.CreateLayer("overlap", srs=sr)

            # Add attributes
            field_defn = ogr.FieldDefn("img1", ogr.OFTString)
            layer.CreateField(field_defn)
            field_defn = ogr.FieldDefn("img2", ogr.OFTString)
            layer.CreateField(field_defn)
            field_defn = ogr.FieldDefn("buffer", ogr.OFTInteger)
            layer.CreateField(field_defn)

            # Create feature and set attributes
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("img1", filename1)
            feature.SetField("img2", filename2)
            feature.SetField("buffer", buffer_size)
            feature.SetGeometry(polygon)
            layer.CreateFeature(feature)

            # Cleanup
            feature    = None
            ds         = None
            buffer_msg = f" with buffer={buffer_size}" if buffer_size > 0 else ""
            print(f"Saved overlap polygon{buffer_msg} to {file_path}")
        # Return the polygon and spatial reference
        return polygon, sr

    def _cut_image(self, img_dict, overlap_shape, overlap_sr, save_overlap=False, save_path=None, no_data_value=None):
        """
        Crop image using gdalwarp with cutline
        """
        # Create a temporary shapefile for the cutline
        temp_shapefile = "/vsimem/temp_cutline.json"
    
        # Clone and transform shape if needed
        if overlap_shape is None:
            print("No overlap shape provided")   
            return None
        shape_to_use  = overlap_shape.Clone()

        # Transform shape if needed
        img_sr        = osr.SpatialReference()
        img_sr.ImportFromWkt(img_dict["gdalImg"].GetProjection())
        if not img_sr.IsSame(overlap_sr):
            transform = osr.CoordinateTransformation(overlap_sr, img_sr)
            shape_to_use.Transform(transform)
        
        # Write shape to temporary file
        driver  = ogr.GetDriverByName('GeoJSON')
        ds      = driver.CreateDataSource(temp_shapefile)
        layer   = ds.CreateLayer("cutline", img_sr)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(shape_to_use)
        layer.CreateFeature(feature)
        # Close dataset
        ds      = None                                           

        if no_data_value is None:
            # If not specified, try to get from source image
            no_data_value = img_dict["gdalImg"].GetRasterBand(1).GetNoDataValue()

            # If source has no NoData value, use default based on data type
            if no_data_value is None:
                data_type = img_dict["gdalImg"].GetRasterBand(1).DataType
                if data_type == gdal.GDT_Byte:
                    no_data_value = 0
                elif data_type in [gdal.GDT_UInt16, gdal.GDT_Int16]:
                    no_data_value = -9999
                else:
                    no_data_value = -9999.0

        # Set up warp options
        warp_options = gdal.WarpOptions(
            cutlineDSName = temp_shapefile,
            cropToCutline = True,
            dstNodata     = no_data_value,
            srcNodata     = img_dict["gdalImg"].GetRasterBand(1).GetNoDataValue(),  # Source NoData if available
            resampleAlg   = gdal.GRA_NearestNeighbour,
            multithread   = True,
            options=['CUTLINE_ALL_TOUCHED=TRUE']
        )

        # Create output destination path for GDAL Warp
        if save_overlap and save_path:
            base_name = os.path.splitext(os.path.basename(img_dict["filepath"]))[0]
            dest_path = os.path.join(save_path, f"{base_name}_cut.tif")
        else:
            # Use in-memory output
            dest_path = '/vsimem/temp_output.tif'

        try:
            # Perform the warp operation with explicit destination path
            result = gdal.Warp(dest_path, img_dict["gdalImg"], options=warp_options)
            
            if result is None:
                print(f"Error: GDAL Warp operation failed for {img_dict['filepath']}")
                return None

            # If we're not saving, we need to clone the dataset before it gets deleted
            if not (save_overlap and save_path):
                driver  = gdal.GetDriverByName('MEM')
                out_img = driver.CreateCopy('', result)
                # Close the temporary file
                result = None
                gdal.Unlink(dest_path)
            else:
                out_img = result
                print(f"Saved warped image to {dest_path}")

            for i in range(1, out_img.RasterCount + 1):
                out_img.GetRasterBand(i).SetNoDataValue(no_data_value)
        finally:
            # Clean up temporary shapefile
            gdal.Unlink(temp_shapefile)
        return out_img
    
    def _get_overlap_resampled(self, img_name1, img_name2, save_overlap=False, save_path=None, no_data_value=np.nan):
        """
        Extracts the exact overlapping region between two images.

        INPUT: 
          img_name1:        Image name of the first image as string (same as key() values in self.images)
          img_name2:        Image name of the second image as string (same as key() values in self.images)
          save_overlap:     Boolean to save the overlap as a new geotiff file (default: False)
          save_path:        Path to save the overlap geotiff files (default: None)
          no_data_value:    No-data value for the output arrays (default: np.nan)

        OUTPUT:
          Two overlapping datasets with identical bounds and dimensions.
          The datasets contain geotransform and projection information as attributes.

        """
        # Get the gdal dataset for both images
        ds_1 = self.images[img_name1]["gdalImg"]
        ds_2 = self.images[img_name2]["gdalImg"]

        # Get geotransforms for both images
        gt1 = ds_1.GetGeoTransform()
        gt2 = ds_2.GetGeoTransform()

        # Get the image width and height for both images
        width1   = ds_1.RasterXSize
        height1  = ds_1.RasterYSize
        width2   = ds_2.RasterXSize
        height2  = ds_2.RasterYSize

        # Check if the images have the same number of bands => Raise an error if they don't
        num_bands = ds_1.RasterCount
        if ds_2.RasterCount != num_bands:
            raise ValueError(f"Images have different number of bands: {num_bands} vs {ds_2.RasterCount}")

        # Get the image corners in pixel coordinates for the first image
        corners1 = [
            (0, 0),
            (width1, 0),
            (width1, height1),
            (0, height1)
        ]

        # Get the image corners in pixel coordinates for the second image
        corners2 = [
            (0, 0),
            (width2, 0),
            (width2, height2),
            (0, height2)
        ]

        # Transform corners to world coordinates for image 1
        world_corners1 = []
        for x, y in corners1:
            world_x = gt1[0] + x * gt1[1] + y * gt1[2]
            world_y = gt1[3] + x * gt1[4] + y * gt1[5]
            world_corners1.append((world_x, world_y))

        # Transform corners to world coordinates for image 2
        world_corners2 = []
        for x, y in corners2:
            world_x = gt2[0] + x * gt2[1] + y * gt2[2]
            world_y = gt2[3] + x * gt2[4] + y * gt2[5]
            world_corners2.append((world_x, world_y))

        # Get bounds for both images in world coordinates
        corner_xs1, corner_ys1 = zip(*world_corners1)
        corner_xs2, corner_ys2 = zip(*world_corners2)

        # Calculate original image extents
        img1_west  = min(corner_xs1)
        img1_east  = max(corner_xs1)
        img1_south = min(corner_ys1)
        img1_north = max(corner_ys1)
        
        img2_west  = min(corner_xs2)
        img2_east  = max(corner_xs2)
        img2_south = min(corner_ys2)
        img2_north = max(corner_ys2)

        # Calculate overlap bounds
        exact_overlap_west  = max(min(corner_xs1), min(corner_xs2))
        exact_overlap_east  = min(max(corner_xs1), max(corner_xs2))
        exact_overlap_south = max(min(corner_ys1), min(corner_ys2))
        exact_overlap_north = min(max(corner_ys1), max(corner_ys2))

        # Check if there is an overlap between the images when not considering the edge frame
        if exact_overlap_east <= exact_overlap_west or exact_overlap_north <= exact_overlap_south:
            print("No overlap between images")
            return None, None

        pixel_width1  = abs(gt1[1])   # Positive pixel width
        pixel_height1 = abs(gt1[5])   # Positive pixel height
        pixel_width2  = abs(gt2[1])   # Positive pixel width
        pixel_height2 = abs(gt2[5])   # Positive pixel height
    
        # Use the finer resolution of the two images
        pixel_width  = min(pixel_width1, pixel_width2)
        pixel_height = min(pixel_height1, pixel_height2)

        # Calculate expanded overlap bounds with the frame
        overlap_west  = exact_overlap_west
        overlap_east  = exact_overlap_east
        overlap_south = exact_overlap_south
        overlap_north = exact_overlap_north

        # Calculate dimensions in pixels, ensuring whole pixels
        width_meters  = overlap_east  - overlap_west
        height_meters = overlap_north - overlap_south

        # Use an integer number of pixels
        grid_width  = max(1, int(np.ceil(width_meters / pixel_width)))
        grid_height = max(1, int(np.ceil(height_meters / pixel_height)))

        # Adjust bounds to ensure exact pixel alignment
        adjusted_width  = grid_width * pixel_width
        adjusted_height = grid_height * pixel_height

        # Center the adjusted region on the original overlap
        width_diff  = adjusted_width  - width_meters
        height_diff = adjusted_height - height_meters

        # Ensure perfectly aligned output by adjusting bounds
        overlap_west -= width_diff / 2
        overlap_east += width_diff / 2
        overlap_south -= height_diff / 2
        overlap_north += height_diff / 2

        # Create new geotransform for the output
        output_geotransform = (
            overlap_west,             # Left edge (X origin)
            pixel_width,              # Pixel width 
            0,                        # X skew
            overlap_north,            # Top edge (Y origin)
            0,                        # Y skew
            -pixel_height             # Negative pixel height
        )
        
        # Calculate the coordinates of each pixel in the output
        x_coords = np.linspace(
            overlap_west + pixel_width/2,     # Start at half-pixel in
            overlap_east - pixel_width/2,     # End at half-pixel in
            grid_width
        )
        
        y_coords = np.linspace(
            overlap_north - pixel_height/2,   # Start at half-pixel in
            overlap_south + pixel_height/2,   # End at half-pixel in
            grid_height
        )

        # Create meshgrid for pixel coordinates in the overlap region (world to pixel)
        X, Y   = np.meshgrid(x_coords, y_coords)

        # Get pixel coordinates of the overlap region for both images in local pixel coordinates
        pixel_x1, pixel_y1 = self._world_to_pixel(X, Y, gt1)
        pixel_x2, pixel_y2 = self._world_to_pixel(X, Y, gt2)

        # Create masks for valid pixels for image1 and image2
        mask1 = ((pixel_x1 >= 0) & (pixel_x1 < width1) & 
                 (pixel_y1 >= 0) & (pixel_y1 < height1))
        mask2 = ((pixel_x2 >= 0) & (pixel_x2 < width2) & 
                 (pixel_y2 >= 0) & (pixel_y2 < height2))

        # Create world coordinate masks for the exact overlap and frame regions
        in_exact_overlap = (
            (X >= exact_overlap_west) & (X <= exact_overlap_east) &
            (Y >= exact_overlap_south) & (Y <= exact_overlap_north)
        )

        combined_mask = mask1 & mask2

        # Initialize output arrays with no_data_value
        output1 = np.full((num_bands, grid_height, grid_width), 
                          no_data_value, dtype=np.float32)
        output2 = np.full_like(output1, no_data_value)

        for band_idx in range(num_bands):
            band1 = ds_1.GetRasterBand(band_idx + 1)  # GDAL bands are 1-based
            band2 = ds_2.GetRasterBand(band_idx + 1)

            # Read the band data
            data1 = band1.ReadAsArray()
            data2 = band2.ReadAsArray()

            # Get the no-data value for the band
            band1_no_data_value = band1.GetNoDataValue()
            band2_no_data_value = band2.GetNoDataValue()

            # Process image 1
            #px1 = np.round(pixel_x1[combined_mask]).astype(int)
            #py1 = np.round(pixel_y1[combined_mask]).astype(int)
            
            # Initialize with no_data_value
            output1[band_idx] = no_data_value
            
            # Only process pixels where mask1 is True
            mask1_valid = mask1 & combined_mask
            px1_valid = np.round(pixel_x1[mask1_valid]).astype(int)
            py1_valid = np.round(pixel_y1[mask1_valid]).astype(int)
            
            # Get values from valid positions
            vals1 = data1[py1_valid, px1_valid]
            
            # Apply no-data check if needed
            if band1_no_data_value is not None:
                vals1_valid = vals1 != band1_no_data_value
                vals1[~vals1_valid] = no_data_value
            
            # Assign values to output using boolean indexing
            valid_y, valid_x = np.where(mask1_valid)
            output1[band_idx, valid_y, valid_x] = vals1
            
            # Process image 2 (same approach)
            mask2_valid = mask2 & combined_mask
            px2_valid = np.round(pixel_x2[mask2_valid]).astype(int)
            py2_valid = np.round(pixel_y2[mask2_valid]).astype(int)
            
            # Get values from valid positions
            vals2 = data2[py2_valid, px2_valid]
            
            # Apply no-data check if needed
            if band2_no_data_value is not None:
                vals2_valid = vals2 != band2_no_data_value
                vals2[~vals2_valid] = no_data_value
            
            # Assign values to output using boolean indexing
            valid_y, valid_x = np.where(mask2_valid)
            output2[band_idx, valid_y, valid_x] = vals2

        # Create masked arrays
        output1 = np.ma.masked_equal(output1, no_data_value)
        output2 = np.ma.masked_equal(output2, no_data_value)

        # Store precise georeferencing information
        output1.geotransform = output_geotransform
        output1.projection   = ds_1.GetProjection()

        output2.geotransform = output_geotransform  # Same geotransform for both
        output2.projection   = ds_2.GetProjection()

        if save_overlap:
            # Save the overlap as a new geotiff
            if save_path is None:
                # Use parent folder
                save_path = os.path.dirname(img_name1)

            base_name1 = os.path.splitext(os.path.basename(img_name1))[0]
            base_name2 = os.path.splitext(os.path.basename(img_name2))[0]

            overlap_path1 = os.path.join(save_path, f"{base_name1}_overlap2.tif")
            overlap_path2 = os.path.join(save_path, f"{base_name2}_overlap1.tif")

            self._save_overlap_geotiff(output1, overlap_path1)
            self._save_overlap_geotiff(output2, overlap_path2)
            print(f"Saved overlap geotiffs to {overlap_path2} and {overlap_path1}")
        return output1, output2

    def _apply_shift(self, image, x_shift, y_shift):
        """
        Apply a shift to an image and save it to a new file.
        - The shift is calculated in pixels.
        => Calculate the shift in coordinates and apply it to the geotransform.
        """
        # Get the gdal dataset for the image
        ds = image["gdalImg"]

        if x_shift == 0 and y_shift == 0:
            # Save the image without any changes
            img_name = os.path.basename(image["filepath"])
            img_name = img_name.replace(".tif", "_ortho.tif")
            self._save_image(img_name, gdal_img=ds)
            return
        
        # Create a copy of the dataset
        driver = gdal.GetDriverByName("GTiff")
        ds_copy = driver.CreateCopy("/vsimem/temp.tif", ds, 0)

        # Calculate the shift in world coordinates
        x_shift_world = x_shift * ds.GetGeoTransform()[1]
        y_shift_world = y_shift * ds.GetGeoTransform()[5]

        # Apply the shift to the geotransform
        new_gt = list(ds_copy.GetGeoTransform())
        new_gt[0] += x_shift_world
        new_gt[3] += y_shift_world
        ds_copy.SetGeoTransform(new_gt)

        # Get the image name
        img_name = os.path.basename(image["filepath"])
        img_name = img_name.replace(".tif", "_ortho.tif")
        # Save the shifted image
        self._save_image(img_name, gdal_img=ds_copy)
        gdal.Unlink("/vsimem/temp.tif")

    def _save_image(self, img_name, gdal_img, extension=".tif"):
        """
        Save an image to a new file.
        """
        folder_name = self.config["MISSION"]["orthorectification_folder"]

        # image_path
        image_path = os.path.join(folder_name, img_name)
        if extension != ".tif":
            image_path = image_path.replace(".tif", extension)

        # Save the image
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(image_path, 
                               xsize=gdal_img.RasterXSize,
                               ysize=gdal_img.RasterYSize,
                               bands=gdal_img.RasterCount,
                               eType=gdal.GDT_Byte)

        out_ds.SetGeoTransform(gdal_img.GetGeoTransform())
        out_ds.SetProjection(gdal_img.GetProjection())

        for i in range(1, gdal_img.RasterCount + 1):
            band     = gdal_img.GetRasterBand(i)  # GDAL bands are 1-based
            out_band = out_ds.GetRasterBand(i)
            out_band.WriteArray(band.ReadAsArray())
            out_band.FlushCache()
            out_band.ComputeStatistics(False)

        out_ds = None  # Close the dataset
        return True

    @staticmethod
    def _world_to_pixel(x, y, geotransform):
        det     = geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4]
        if np.isclose(det, 0):
            raise ValueError("Invalid geotransform: Determinant is zero")

        pixel_x = (geotransform[5] * (x - geotransform[0]) - 
                  geotransform[2] * (y - geotransform[3])) / det
        pixel_y = (-geotransform[4] * (x - geotransform[0]) + 
                  geotransform[1] * (y - geotransform[3])) / det
        return np.round(pixel_x).astype(int), np.round(pixel_y).astype(int)

    @staticmethod
    def _save_overlap_geotiff(geo_array, output_path):
        """Save an overlap region as a new geotiff file."""
        data = geo_array.data
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(-9999)

        if len(data.shape) == 2:
            num_bands = 1
            height, width = data.shape
            data = data.reshape(1, height, width)
        else:
            num_bands, height, width = data.shape

        driver = gdal.GetDriverByName("GTiff")

        # Determine the data type
        if data.dtype == np.float64 or data.dtype == np.float32:
            gdal_dtype = gdal.GDT_Float32
        elif data.dtype == np.int32:
            gdal_dtype = gdal.GDT_Int32
        elif data.dtype == np.uint16:
            gdal_dtype = gdal.GDT_UInt16
        elif data.dtype == np.int16:
            gdal_dtype = gdal.GDT_Int16
        elif data.dtype == np.uint8:
            gdal_dtype = gdal.GDT_Byte
        else:
            gdal_dtype = gdal.GDT_Float32

        try:
            # Create the output dataset with a single band
            out_ds = driver.Create(
                output_path,
                width,
                height,
                num_bands,
                gdal_dtype
            )

            out_ds.SetGeoTransform(geo_array.geotransform)
            out_ds.SetProjection(geo_array.projection)

            #  Write each band
            for band_idx in range(num_bands):
                out_band = out_ds.GetRasterBand(band_idx + 1)
                out_band.SetNoDataValue(-9999)
                out_band.WriteArray(data[band_idx, :, :])
                out_band.FlushCache()
                out_band.ComputeStatistics(False)
            
        except Exception as e:
            raise Exception(f"Error saving GeoTIFF: {str(e)}")
        finally:
            out_ds = None  # Close the dataset
        return True
    
    def _shift_opticalFlow(self, ref_img, target_img, band=0, verbose=False):
        """
        Calculate local shift between two overlapping images to properly align them, using optical flow.
        """
        if ref_img is None or target_img is None:
            if verbose:
                print("Error: One of the input images is None")
            return 0, 0, None
        
        # For GDAL datasets, check the band count
        ref_band_count    = ref_img.RasterCount
        target_band_count = target_img.RasterCount
    
        if band >= ref_band_count:
            raise ValueError(f"Band index {band} out of range: ref_img has {ref_band_count} bands")
        if band >= target_band_count:
            raise ValueError(f"Band index {band} out of range: target_img has {target_band_count} bands")
        
        # Read the data from the specified band (GDAL bands are 1-based)
        ref_band    = ref_img.GetRasterBand(band + 1)
        target_band = target_img.GetRasterBand(band + 1)
        
        ref_data    = ref_band.ReadAsArray()
        target_data = target_band.ReadAsArray()
    
        # Print the shapes of the arrays to debug
        if verbose:
            print(f"Reference image shape: {ref_data.shape}")
            print(f"Target image shape: {target_data.shape}")
    
        # Get NoData values
        ref_nodata    = ref_band.GetNoDataValue()
        target_nodata = target_band.GetNoDataValue()
    
        if ref_data.shape != target_data.shape:
            if verbose:
                print(f"Images have different dimensions. Resampling to match.")
            #print("Warning: Images have different dimensions. Resampling to match.")
            
            # Choose a common size (the smaller of the two to avoid artifacts)
            common_height = min(ref_data.shape[0], target_data.shape[0])
            common_width  = min(ref_data.shape[1], target_data.shape[1])
            
            # Use INTER_AREA for downsampling and INTER_CUBIC for upsampling
            # This generally produces better results than LINEAR interpolation
            if common_width < ref_data.shape[1] or common_height < ref_data.shape[0]:
                ref_interp = cv2.INTER_AREA
            else:
                ref_interp = cv2.INTER_CUBIC
            
            if common_width < target_data.shape[1] or common_height < target_data.shape[0]:
                target_interp = cv2.INTER_AREA
            else:
                target_interp = cv2.INTER_CUBIC
            # Resize both images to the common size
            ref_ds    = cv2.resize(ref_data, (common_width, common_height), interpolation=ref_interp)
            target_ds = cv2.resize(target_data, (common_width, common_height), interpolation=target_interp)

            # Create masks after resizing
            ref_mask    = np.ones_like(ref_ds, dtype=bool)
            target_mask = np.ones_like(target_ds, dtype=bool)
            
            # Handle NoData values
            if ref_nodata is not None:
                # Apply a threshold to account for interpolation effects
                ref_mask = np.abs(ref_ds - ref_nodata) > 2.0
            elif np.issubdtype(ref_ds.dtype, np.floating):
                ref_mask = ~np.isnan(ref_ds)
                
            if target_nodata is not None:
                target_mask = np.abs(target_ds - target_nodata) > 2.0
            elif np.issubdtype(target_ds.dtype, np.floating):
                target_mask = ~np.isnan(target_ds)
        else:
            # Images have the same dimensions, proceed normally
            ref_ds    = np.copy(ref_data)
            target_ds = np.copy(target_data)
            
            # Create masks for valid pixels
            ref_mask    = np.ones_like(ref_ds, dtype=bool)
            target_mask = np.ones_like(target_ds, dtype=bool)
            
            if ref_nodata is not None:
                ref_mask = ref_ds != ref_nodata
            elif np.issubdtype(ref_ds.dtype, np.floating):
                ref_mask = ~np.isnan(ref_ds)
                
            if target_nodata is not None:
                target_mask = target_ds != target_nodata
            elif np.issubdtype(target_ds.dtype, np.floating):
                target_mask = ~np.isnan(target_ds)
        
        # Now ref_mask and target_mask should have the same shape as ref_ds and target_ds
        assert ref_mask.shape    == ref_ds.shape,    f"Mask shape mismatch: {ref_mask.shape}    vs {ref_ds.shape}"
        assert target_mask.shape == target_ds.shape, f"Mask shape mismatch: {target_mask.shape} vs {target_ds.shape}"
        
        valid_mask = ref_mask & target_mask
        
        # Skip calculation if there's not enough valid data
        valid_percentage = np.sum(valid_mask) / valid_mask.size
        if valid_percentage < 0.25:
            if verbose:
                print(f"Warning: Not enough valid data in overlap ({valid_percentage:.1%}). Skipping optical flow.")
            return 0, 0, None
    
        # Replace invalid values with the mean of valid values (better than zeros)
        # This helps avoid artificial edges at mask boundaries
        if np.any(ref_mask):
            ref_mean = np.mean(ref_ds[ref_mask])
            ref_ds[~ref_mask] = ref_mean
        else:
            ref_ds[~ref_mask] = 0
    
        if np.any(target_mask):
            target_mean = np.mean(target_ds[target_mask])
            target_ds[~target_mask] = target_mean
        else:
            target_ds[~target_mask] = 0

        # Convert to correct format for OpenCV
        ref_ds    = ref_ds.astype(np.float32)
        target_ds = target_ds.astype(np.float32)
        
        # Enhance contrast in the valid regions for better feature matching
        if np.any(ref_mask):
            # Calculate statistics only from valid data
            ref_valid        = ref_ds[ref_mask]
            ref_min, ref_max = np.percentile(ref_valid, [2, 98])  # Use percentiles to avoid outliers
            if ref_max > ref_min:
                # Improved contrast enhancement with histogram equalization
                ref_ds_norm = np.zeros_like(ref_ds)
                ref_ds_norm[ref_mask]  = np.clip((ref_ds[ref_mask] - ref_min) / (ref_max - ref_min) * 255, 0, 255)
                ref_ds_norm[~ref_mask] = 0

                # Convert to uint8 for histogram equalization
                ref_ds_uint8 = ref_ds_norm.astype(np.uint8)
                
                # Apply histogram equalization only to valid regions
                # Create a mask for histogram equalization
                mask_uint8 = ref_mask.astype(np.uint8) * 255

                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                # This often works better than simple histogram equalization
                clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                ref_ds_eq = clahe.apply(ref_ds_uint8)

                # Convert back to float32
                ref_ds = ref_ds_eq.astype(np.float32)

        if np.any(target_mask):
            # Calculate statistics only from valid data
            target_valid           = target_ds[target_mask]
            target_min, target_max = np.percentile(target_valid, [2, 98])
            if target_max > target_min:
                target_ds_norm = np.zeros_like(target_ds)
                target_ds_norm[target_mask] = np.clip((target_ds[target_mask] - target_min) / (target_max - target_min) * 255, 0, 255)
                target_ds_norm[~target_mask] = 0
                
                target_ds_uint8 = target_ds_norm.astype(np.uint8)
                mask_uint8      = target_mask.astype(np.uint8) * 255
                
                clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                target_ds_eq = clahe.apply(target_ds_uint8)
                
                target_ds = target_ds_eq.astype(np.float32)

        # Apply Gaussian blur to reduce noise
        ref_ds    = cv2.GaussianBlur(ref_ds, (3, 3), 0)
        target_ds = cv2.GaussianBlur(target_ds, (3, 3), 0)
    
        # Continue with the rest of your existing optical flow code...
        try:
            # Try multiple parameter combinations and pick the most consistent result
            flow_results = []
            confidence_scores = []
            
            # Parameter combinations for optical flow
            param_sets = [
                # Standard parameters
                {
                    'pyr_scale': 0.5,
                    'levels': 5,
                    'winsize': 21,
                    'iterations': 10,
                    'poly_n': 7,
                    'poly_sigma': 1.5,
                },
                # More detailed parameters for small shifts
                {
                    'pyr_scale': 0.8,  # Slower scale reduction preserves more detail
                    'levels': 3,
                    'winsize': 15,
                    'iterations': 15,
                    'poly_n': 5,
                    'poly_sigma': 1.1,
                },
                # Parameters for larger shifts
                {
                    'pyr_scale': 0.5,
                    'levels': 6,  # More pyramid levels for larger motions
                    'winsize': 31,  # Larger window for more global motion estimation
                    'iterations': 8,
                    'poly_n': 5,
                    'poly_sigma': 1.5,
                }
            ]
            
            for params in param_sets:
                # Compute optical flow with current parameter set
                flow = cv2.calcOpticalFlowFarneback(
                    ref_ds, target_ds, None,
                    pyr_scale=params['pyr_scale'],
                    levels=params['levels'],
                    winsize=params['winsize'],
                    iterations=params['iterations'], 
                    poly_n=params['poly_n'],
                    poly_sigma=params['poly_sigma'],
                    flags=0
                )
                
                # Extract flow components
                x_flow = flow[:, :, 0]
                y_flow = flow[:, :, 1]
                
                # More sophisticated flow analysis using histogram-based approach
                # Only consider valid regions for flow statistics
                x_valid = x_flow[valid_mask]
                y_valid = y_flow[valid_mask]
                
                # Calculate magnitude and direction
                flow_magnitude = np.sqrt(x_valid**2 + y_valid**2)
                
                # Use histogram analysis to find the dominant motion
                hist_x, bins_x = np.histogram(x_valid, bins=50)
                hist_y, bins_y = np.histogram(y_valid, bins=50)
                
                # Find the bin with most counts - represents dominant motion component
                dominant_x_bin = np.argmax(hist_x)
                dominant_y_bin = np.argmax(hist_y)
                
                # Get the center value of that bin
                x_shift = (bins_x[dominant_x_bin] + bins_x[dominant_x_bin + 1]) / 2
                y_shift = (bins_y[dominant_y_bin] + bins_y[dominant_y_bin + 1]) / 2
                
                # Alternative: use percentile-based approach (more robust than median)
                # Get values at 40th and 60th percentiles
                x_40p = np.percentile(x_valid, 40)
                x_60p = np.percentile(x_valid, 60)
                y_40p = np.percentile(y_valid, 40)
                y_60p = np.percentile(y_valid, 60)
                
                # Use average of these percentiles
                x_shift_alt = (x_40p + x_60p) / 2
                y_shift_alt = (y_40p + y_60p) / 2
                
                # Decide which approach to use based on histogram peak sharpness
                x_peak_ratio = hist_x[dominant_x_bin] / np.mean(hist_x)
                y_peak_ratio = hist_y[dominant_y_bin] / np.mean(hist_y)
                
                # If histogram has a sharp peak, use it, otherwise use percentile approach
                if x_peak_ratio > 2.0:
                    final_x_shift = x_shift
                else:
                    final_x_shift = x_shift_alt
                    
                if y_peak_ratio > 2.0:
                    final_y_shift = y_shift
                else:
                    final_y_shift = y_shift_alt
                
                # Calculate flow statistics 
                mean_magnitude = np.mean(flow_magnitude)
                std_magnitude = np.std(flow_magnitude)
                
                # Flow consistency measurement (how uniform is the flow)
                # Lower deviation relative to magnitude means more consistent flow
                consistency_ratio = std_magnitude / (mean_magnitude + 1e-6)
                
                # Improved confidence assessment
                confidence = 1.0
                
                # Penalize inconsistent flow
                if consistency_ratio > 0.8:
                    confidence *= (1.0 - (consistency_ratio - 0.8) / 0.8)
                
                # Reward strong, consistent flow patterns
                if mean_magnitude > 2.0 and consistency_ratio < 0.6:
                    confidence *= 1.2
                    confidence = min(confidence, 1.0)
                
                # Check flow direction consistency
                flow_x_std = np.std(x_valid)
                flow_y_std = np.std(y_valid)
                
                # If variation is too high compared to shift magnitude, reduce confidence
                if flow_x_std > 2 * abs(final_x_shift) or flow_y_std > 2 * abs(final_y_shift):
                    confidence *= 0.8
                
                if verbose:
                    print(f"Parameters set {param_sets.index(params) + 1}:")
                    print(f"  Shift: x={final_x_shift:.2f}, y={final_y_shift:.2f}")
                    print(f"  Mean magnitude: {mean_magnitude:.2f}, StdDev: {std_magnitude:.2f}")
                    print(f"  Consistency ratio: {consistency_ratio:.2f}")
                    print(f"  Confidence: {confidence:.2f}")
                
                # Store results
                flow_results.append((final_x_shift, final_y_shift))
                confidence_scores.append(confidence)
                
            # Combine results from all parameter sets, weighted by confidence
            if len(flow_results) > 0:
                # Convert to numpy arrays
                shifts = np.array(flow_results)
                confidences = np.array(confidence_scores)
                
                # Normalize confidences to sum to 1
                if np.sum(confidences) > 0:
                    weights = confidences / np.sum(confidences)
                else:
                    weights = np.ones_like(confidences) / len(confidences)
                
                # Weighted average of shifts
                x_shift = np.sum(shifts[:, 0] * weights)
                y_shift = np.sum(shifts[:, 1] * weights)
                
                # Get best confidence score
                best_confidence = np.max(confidences)
                
                # Calculate overall statistics from best parameter set
                best_idx = np.argmax(confidences)
                
                if verbose:
                    print(f"Final weighted shift: x={x_shift:.2f}, y={y_shift:.2f}")
                    print(f"Best confidence: {best_confidence:.2f}")
            else:
                # Fallback if all parameter sets failed
                x_shift = 0
                y_shift = 0
                best_confidence = 0
                mean_magnitude = 0
                std_magnitude = 0
                if verbose:
                    print("All parameter sets failed to produce valid flow")
    
            # IMPORTANT CHANGE: Do not zero out small shifts by default
            # Only zero out extremely small shifts if confidence is very low
            if best_confidence < 0.3 and abs(x_shift) < 0.5 and abs(y_shift) < 0.5:
                if verbose:
                    print("Very low confidence and tiny shift, zeroing out")
                x_shift = 0
                y_shift = 0
    
            ## Convert the shift to real-world coordinates
            #if geotransform_target is None:
            #    raise ValueError("Target image geotransform is missing")
            
            # Apply the geotransform to calculate actual shifts in meters
            x_shift_m = x_shift * abs(target_img.GetGeoTransform()[1])
            y_shift_m = y_shift * abs(target_img.GetGeoTransform()[5])
    
            if verbose:
                print(f"Final shift in meters: x={x_shift_m:.4f}m, y={y_shift_m:.4f}m")
                print(f"Confidence: {best_confidence:.4f}")
    
            # Return the shift values and confidence statistics
            # Note: We're no longer ignoring small shifts by default
            statistics = (best_confidence, mean_magnitude, std_magnitude)
            return x_shift_m, y_shift_m, statistics

        except cv2.error as e:
            print(f"OpenCV error: {str(e)}")
            print(f"Image shapes - ref: {ref_ds.shape}, target: {target_ds.shape}")
            print(f"Value ranges - ref: {np.min(ref_ds)}-{np.max(ref_ds)}, target: {np.min(target_ds)}-{np.max(target_ds)}")
            return 0, 0, None
        except Exception as e:
            print(f"Error in optical flow calculation: {str(e)}")
            return 0, 0, None

    def _shift_opticalFlow_multiscale(self, ref_img, target_img, band=0, verbose=False):
        """
        Calculate local shift between two overlapping images using a multi-scale approach.
        This handles larger shifts more effectively by working at different resolutions.
        """
        tuning_val_percentage = 0.25  # Minimum valid data percentage for optical flow


        if ref_img is None or target_img is None:
            if verbose:
                print("Error: One of the input images is None")
            return 0, 0, None
        
        # For GDAL datasets, check the band count
        ref_band_count    = ref_img.RasterCount
        target_band_count = target_img.RasterCount
    
        if band >= ref_band_count:
            raise ValueError(f"Band index {band} out of range: ref_img has {ref_band_count} bands")
        if band >= target_band_count:
            raise ValueError(f"Band index {band} out of range: target_img has {target_band_count} bands")
        
        # Read the data from the specified band (GDAL bands are 1-based)
        ref_band    = ref_img.GetRasterBand(band + 1)
        target_band = target_img.GetRasterBand(band + 1)
        
        ref_data    = ref_band.ReadAsArray()
        target_data = target_band.ReadAsArray()
    
        # Print the shapes of the arrays to debug
        if verbose:
            print(f"Reference image shape: {ref_data.shape}")
            print(f"Target image shape: {target_data.shape}")
    
        # Get NoData values
        ref_nodata    = ref_band.GetNoDataValue()
        target_nodata = target_band.GetNoDataValue()
    
        if ref_data.shape != target_data.shape:
            if verbose:
                print(f"Images have different dimensions. Resampling to match.")
            
            # Choose a common size (the smaller of the two to avoid artifacts)
            common_height = min(ref_data.shape[0], target_data.shape[0])
            common_width  = min(ref_data.shape[1], target_data.shape[1])
            
            # Use INTER_AREA for downsampling and INTER_CUBIC for upsampling
            if common_width < ref_data.shape[1] or common_height < ref_data.shape[0]:
                ref_interp = cv2.INTER_AREA
            else:
                ref_interp = cv2.INTER_CUBIC
            
            if common_width < target_data.shape[1] or common_height < target_data.shape[0]:
                target_interp = cv2.INTER_AREA
            else:
                target_interp = cv2.INTER_CUBIC
                
            # Resize both images to the common size
            ref_ds    = cv2.resize(ref_data, (common_width, common_height), interpolation=ref_interp)
            target_ds = cv2.resize(target_data, (common_width, common_height), interpolation=target_interp)
        else:
            # Images have the same dimensions, proceed normally
            ref_ds    = np.copy(ref_data)
            target_ds = np.copy(target_data)
        
        # Create masks for valid pixels
        ref_mask    = np.ones_like(ref_ds, dtype=bool)
        target_mask = np.ones_like(target_ds, dtype=bool)
        
        if ref_nodata is not None:
            ref_mask = ref_ds != ref_nodata
        elif np.issubdtype(ref_ds.dtype, np.floating):
            ref_mask = ~np.isnan(ref_ds)
            
        if target_nodata is not None:
            target_mask = target_ds != target_nodata
        elif np.issubdtype(target_ds.dtype, np.floating):
            target_mask = ~np.isnan(target_ds)
        
        valid_mask = ref_mask & target_mask
        
        # Skip calculation if there's not enough valid data
        valid_percentage = np.sum(valid_mask) / valid_mask.size
        if valid_percentage < tuning_val_percentage:
            if verbose:
                print(f"Warning: Not enough valid data in overlap ({valid_percentage:.1%}). Skipping optical flow.")
            return 0, 0, None
    
        # Replace invalid values with the mean of valid values
        if np.any(ref_mask):
            ref_mean          = np.mean(ref_ds[ref_mask])
            ref_ds[~ref_mask] = ref_mean
        else:
            ref_ds[~ref_mask] = 0
    
        if np.any(target_mask):
            target_mean             = np.mean(target_ds[target_mask])
            target_ds[~target_mask] = target_mean
        else:
            target_ds[~target_mask] = 0
    
        # Convert to correct format for OpenCV
        ref_ds    = ref_ds.astype(np.float32)
        target_ds = target_ds.astype(np.float32)
        
        # Enhance contrast for better feature matching
        if np.any(ref_mask):
            ref_valid        = ref_ds[ref_mask]
            ref_min, ref_max = np.percentile(ref_valid, [2, 98])
            if ref_max > ref_min:
                ref_ds_norm            = np.zeros_like(ref_ds)
                ref_ds_norm[ref_mask]  = np.clip((ref_ds[ref_mask] - ref_min) / (ref_max - ref_min) * 255, 0, 255)
                ref_ds_norm[~ref_mask] = 0
                ref_ds_uint8           = ref_ds_norm.astype(np.uint8)
                clahe                  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                ref_ds_eq              = clahe.apply(ref_ds_uint8)
                ref_ds                 = ref_ds_eq.astype(np.float32)
    
        if np.any(target_mask):
            target_valid           = target_ds[target_mask]
            target_min, target_max = np.percentile(target_valid, [2, 98])
            if target_max > target_min:
                target_ds_norm               = np.zeros_like(target_ds)
                target_ds_norm[target_mask]  = np.clip((target_ds[target_mask] - target_min) / (target_max - target_min) * 255, 0, 255)
                target_ds_norm[~target_mask] = 0
                target_ds_uint8              = target_ds_norm.astype(np.uint8)
                clahe                        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                target_ds_eq                 = clahe.apply(target_ds_uint8)
                target_ds                    = target_ds_eq.astype(np.float32)
        
        # Create a pyramid of downsampled images for multi-scale approach
        pyramid_levels     = 4
        ref_pyramid        = [ref_ds]
        target_pyramid     = [target_ds]
        valid_mask_pyramid = [valid_mask]
        
        # Build image pyramids
        for i in range(1, pyramid_levels):
            scale_factor   = 1.0 / (2**i)
            h, w           = ref_ds.shape
            scaled_h       = max(int(h * scale_factor), 32)
            scaled_w       = max(int(w * scale_factor), 32)
            
            # Downsampled images
            ref_down    = cv2.resize(ref_ds, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            target_down = cv2.resize(target_ds, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            mask_down   = cv2.resize(valid_mask.astype(np.uint8), (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            ref_pyramid.append(ref_down)
            target_pyramid.append(target_down)
            valid_mask_pyramid.append(mask_down)
        
        # Start with the coarsest (largest) level and refine progressively
        cumulative_x_shift = 0.0
        cumulative_y_shift = 0.0
        best_confidence    = 0.0
        best_stats         = None
        
        # Apply Gaussian blur to each level
        for i in range(pyramid_levels):
            ref_pyramid[i]    = cv2.GaussianBlur(ref_pyramid[i], (3, 3), 0)
            target_pyramid[i] = cv2.GaussianBlur(target_pyramid[i], (3, 3), 0)
        
        # Process from coarsest to finest
        for i in range(pyramid_levels - 1, -1, -1):
            scale_factor = 2**i
            
            ref_current        = ref_pyramid[i]
            target_current     = target_pyramid[i]
            valid_mask_current = valid_mask_pyramid[i]
            
            # Apply current cumulative shift to the target image (scaled appropriately)
            if i < pyramid_levels - 1:  # Skip first iteration on coarsest level
                h, w           = target_current.shape
                x_shift_scaled = cumulative_x_shift / scale_factor
                y_shift_scaled = cumulative_y_shift / scale_factor
                
                # Create transformation matrix for the shift
                M              = np.float32([[1, 0, x_shift_scaled], [0, 1, y_shift_scaled]])
                target_current = cv2.warpAffine(target_current, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Calculate optical flow at this level
            try:
                # Parameters adjusted for the scale
                win_size   = max(15, int(21 / (i + 1)))
                iterations = max(5, int(10 / (i + 1)))
                
                flow = cv2.calcOpticalFlowFarneback(
                    ref_current, 
                    target_current, 
                    None,
                    pyr_scale  = 0.5,
                    levels     = min(5, pyramid_levels - i),  # Fewer levels at smaller scales
                    winsize    = win_size,
                    iterations = iterations,
                    poly_n     = 5,
                    poly_sigma = 1.1,
                    flags      = 0
                )
                
                # Extract flow components
                x_flow = flow[:, :, 0]
                y_flow = flow[:, :, 1]
                
                # Only consider valid regions for flow statistics
                x_valid = x_flow[valid_mask_current]
                y_valid = y_flow[valid_mask_current]
                
                if len(x_valid) == 0 or len(y_valid) == 0:
                    continue
                    
                # Calculate robust shift statistics
                flow_magnitude = np.sqrt(x_valid**2 + y_valid**2)
                
                # Get dominant shift using both histogram and percentile approaches
                # Histogram approach
                hist_x, bins_x = np.histogram(x_valid, bins=30)
                hist_y, bins_y = np.histogram(y_valid, bins=30)
                
                dominant_x_bin = np.argmax(hist_x)
                dominant_y_bin = np.argmax(hist_y)
                
                x_shift_hist = (bins_x[dominant_x_bin] + bins_x[dominant_x_bin + 1]) / 2
                y_shift_hist = (bins_y[dominant_y_bin] + bins_y[dominant_y_bin + 1]) / 2
                
                # Percentile approach (more robust than median)
                x_perc = np.percentile(x_valid, [40, 50, 60])
                y_perc = np.percentile(y_valid, [40, 50, 60])
                
                x_shift_perc = np.mean(x_perc)
                y_shift_perc = np.mean(y_perc)
                
                # Choose approach based on histogram peak sharpness
                x_peak_ratio = hist_x[dominant_x_bin] / np.mean(hist_x) if np.mean(hist_x) > 0 else 0
                y_peak_ratio = hist_y[dominant_y_bin] / np.mean(hist_y) if np.mean(hist_y) > 0 else 0
                
                if x_peak_ratio > 2.0:
                    level_x_shift = x_shift_hist
                else:
                    level_x_shift = x_shift_perc
                    
                if y_peak_ratio > 2.0:
                    level_y_shift = y_shift_hist
                else:
                    level_y_shift = y_shift_perc
                
                # Scale the shift back up to original image size
                level_x_shift *= scale_factor
                level_y_shift *= scale_factor
                
                # Add to cumulative shift
                cumulative_x_shift += level_x_shift
                cumulative_y_shift += level_y_shift
                
                # Calculate flow statistics
                mean_magnitude = np.mean(flow_magnitude)
                std_magnitude  = np.std(flow_magnitude)
                
                # Flow consistency measurement
                consistency_ratio = std_magnitude / (mean_magnitude + 1e-6)
                
                # Calculate confidence score
                confidence = 1.0
                
                # Penalize inconsistent flow
                if consistency_ratio > 0.8:
                    confidence *= (1.0 - (consistency_ratio - 0.8) / 0.8)
                
                # Reward strong, consistent flow patterns
                if mean_magnitude > 1.0 and consistency_ratio < 0.6:
                    confidence *= 1.2
                    confidence  = min(confidence, 1.0)
                
                # Higher confidence for finer levels
                confidence *= (1.0 + (pyramid_levels - i - 1) * 0.1)
                confidence  = min(confidence, 1.0)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_stats      = (confidence, mean_magnitude, std_magnitude)
                
                if verbose:
                    print(f"Level {i} (scale factor {scale_factor}):")
                    print(f"  Level shift: x={level_x_shift:.2f}, y={level_y_shift:.2f}")
                    print(f"  Cumulative shift: x={cumulative_x_shift:.2f}, y={cumulative_y_shift:.2f}")
                    print(f"  Confidence: {confidence:.2f}")
                
            except Exception as e:
                if verbose:
                    print(f"Error at pyramid level {i}: {str(e)}")
                continue
        
        # Apply the geotransform to calculate actual shifts in meters
        x_shift_m = cumulative_x_shift * abs(target_img.GetGeoTransform()[1])
        y_shift_m = cumulative_y_shift * abs(target_img.GetGeoTransform()[5])
        
        if verbose:
            print(f"Final shift in meters: x={x_shift_m:.4f}m, y={y_shift_m:.4f}m")
            print(f"Confidence: {best_confidence:.4f}")
        
        # If no valid shift was found
        if best_stats is None:
            return 0, 0, None
        
        return x_shift_m, y_shift_m, best_stats

    def _shift_featureBased(self, ref_img, target_img, band=0, verbose=False, visualize=False):
        """
        Calculate shift between two images using feature-based matching (ORB, SIFT, etc.).
        This approach can detect larger shifts than optical flow methods.
        """
        if ref_img is None or target_img is None:
            if verbose:
                print("Error: One of the input images is None")
            return 0, 0, None

        # Read band data
        ref_band    = ref_img.GetRasterBand(band + 1)
        target_band = target_img.GetRasterBand(band + 1)

        ref_data    = ref_band.ReadAsArray()
        target_data = target_band.ReadAsArray()

        # Get NoData values
        ref_nodata    = ref_band.GetNoDataValue()
        target_nodata = target_band.GetNoDataValue()
        
        # Ensure same dimensions
        if ref_data.shape != target_data.shape:
            if verbose:
                print(f"Images have different dimensions. Resampling to match.")
            
            common_height = min(ref_data.shape[0], target_data.shape[0])
            common_width  = min(ref_data.shape[1], target_data.shape[1])
            
            # Use appropriate interpolation method
            if common_width < ref_data.shape[1] or common_height < ref_data.shape[0]:
                ref_interp = cv2.INTER_AREA
            else:
                ref_interp = cv2.INTER_CUBIC
            
            if common_width < target_data.shape[1] or common_height < target_data.shape[0]:
                target_interp = cv2.INTER_AREA
            else:
                target_interp = cv2.INTER_CUBIC
                
            ref_ds    = cv2.resize(ref_data, (common_width, common_height), interpolation=ref_interp)
            target_ds = cv2.resize(target_data, (common_width, common_height), interpolation=target_interp)
        else:
            ref_ds    = np.copy(ref_data)
            target_ds = np.copy(target_data)
        
        # Create masks for valid pixels
        ref_mask    = np.ones_like(ref_ds, dtype=bool)
        target_mask = np.ones_like(target_ds, dtype=bool)
        
        if ref_nodata is not None:
            ref_mask = ref_ds != ref_nodata
        elif np.issubdtype(ref_ds.dtype, np.floating):
            ref_mask = ~np.isnan(ref_ds)
            
        if target_nodata is not None:
            target_mask = target_ds != target_nodata
        elif np.issubdtype(target_ds.dtype, np.floating):
            target_mask = ~np.isnan(target_ds)
        
        valid_mask = ref_mask & target_mask
        
        # Skip calculation if there's not enough valid data
        valid_percentage = np.sum(valid_mask) / valid_mask.size
        if valid_percentage < 0.25:
            if verbose:
                print(f"Warning: Not enough valid data in overlap ({valid_percentage:.1%}). Skipping matching.")
            return 0, 0, None
        
        # Fill invalid pixels with mean value to avoid artificial edges
        if np.any(ref_mask):
            ref_mean          = np.mean(ref_ds[ref_mask])
            ref_ds[~ref_mask] = ref_mean
        else:
            ref_ds[~ref_mask] = 0
        
        if np.any(target_mask):
            target_mean             = np.mean(target_ds[target_mask])
            target_ds[~target_mask] = target_mean
        else:
            target_ds[~target_mask] = 0
        
        # Convert to uint8 for feature detection
        # Apply contrast enhancement
        def normalize_for_features(img, mask):
            if np.any(mask):
                img_valid        = img[mask]
                min_val, max_val = np.percentile(img_valid, [2, 98])
                if max_val > min_val:
                    img_norm       = np.zeros_like(img)
                    img_norm[mask] = np.clip((img[mask] - min_val) / (max_val - min_val) * 255, 0, 255)
                    return img_norm.astype(np.uint8)
            return np.zeros_like(img, dtype=np.uint8)

        ref_uint8    = normalize_for_features(ref_ds, ref_mask)
        target_uint8 = normalize_for_features(target_ds, target_mask)

        # Apply sharpening to enhance features
        kernel       = np.array([[-1, -1, -1], 
                                 [-1,  9, -1], 
                                 [-1, -1, -1]])
        ref_sharp    = cv2.filter2D(ref_uint8, -1, kernel)
        target_sharp = cv2.filter2D(target_uint8, -1, kernel)

        # Apply CLAHE for better contrast
        clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        ref_clahe    = clahe.apply(ref_sharp)
        target_clahe = clahe.apply(target_sharp)

        # Try multiple feature detection methods and use the best result
        methods = ['orb', 'sift', 'akaze']
        best_shift_x     = 0
        best_shift_y     = 0
        best_confidence  = 0
        best_match_count = 0
        best_stats       = None
        
        for method in methods:
            try:
                if method == 'orb':
                    # ORB detector
                    detector = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
                elif method == 'sift':
                    # SIFT detector - better for larger shifts but slower
                    detector = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04)
                elif method == 'akaze':
                    # AKAZE detector - good balance between speed and accuracy
                    detector = cv2.AKAZE_create()
                
                # Find keypoints and descriptors
                kp1, des1 = detector.detectAndCompute(ref_clahe, None)
                kp2, des2 = detector.detectAndCompute(target_clahe, None)
                
                if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                    if verbose:
                        print(f"Not enough features detected with {method}")
                    continue
                
                # Match features
                if method == 'orb':
                    # Use Hamming distance for binary descriptors (ORB)
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                else:
                    # Use L2 distance for float descriptors (SIFT, AKAZE)
                    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                
                matches = matcher.match(des1, des2)
                
                # Filter matches based on distance
                # Sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Take only good matches
                good_ratio       = 0.75 # Ratio of good matches to consider
                num_good_matches = int(len(matches) * good_ratio)
                good_matches     = matches[:num_good_matches]
                
                if len(good_matches) < 10:
                    if verbose:
                        print(f"Not enough good matches found with {method}")
                    continue
                
                # Extract matched keypoints
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                
                # Calculate shifts between matched points
                shifts = dst_pts - src_pts
                
                # Use RANSAC to filter outliers and find dominant shift
                def ransac_shift(shifts, iterations=500, threshold=3.0, min_inliers_ratio=0.3):
                    n_points = shifts.shape[0]
                    if n_points < 5:
                        return np.mean(shifts, axis=0), 0, 0
                    
                    best_inliers = 0
                    best_shift   = np.zeros(2)
                    
                    for _ in range(iterations):
                        # Randomly select a subset of points
                        indices       = np.random.choice(n_points, min(5, n_points), replace=False)
                        sample_shifts = shifts[indices]
                        
                        # Calculate median shift for this sample
                        sample_shift = np.median(sample_shifts, axis=0)
                        
                        # Count inliers
                        distances     = np.sqrt(np.sum((shifts - sample_shift)**2, axis=1))
                        inliers       = distances < threshold
                        inliers_count = np.sum(inliers)
                        
                        if inliers_count > best_inliers:
                            best_inliers = inliers_count
                            best_shift   = np.mean(shifts[inliers], axis=0)
                    
                    inliers_ratio = best_inliers / n_points
                    confidence    = inliers_ratio
                    
                    # Extra check: calculate standard deviation of inlier shifts
                    if best_inliers > 5:
                        inlier_mask = np.sqrt(np.sum((shifts - best_shift)**2, axis=1)) < threshold
                        std_dev     = np.std(shifts[inlier_mask], axis=0)
                        avg_std     = np.mean(std_dev)
                        
                        # Lower confidence if spread is high
                        if avg_std > 5.0:
                            confidence *= (5.0 / avg_std)
                    
                    return best_shift, confidence, best_inliers
                
                # Apply RANSAC to find the dominant shift
                shift, confidence, inliers_count = ransac_shift(shifts)
                shift_x, shift_y                 = shift
                
                # Scale confidence by number of matches and method reliability
                method_reliability  = {'sift': 1.0, 'akaze': 0.9, 'orb': 0.8}
                adjusted_confidence = confidence * method_reliability.get(method, 0.7)
                
                # Give bonus to methods with more matches (up to a limit)
                match_ratio          = min(1.0, len(good_matches) / 200)
                adjusted_confidence *= (0.5 + 0.5 * match_ratio)
                
                # Update best result if this method performed better
                if adjusted_confidence > best_confidence and inliers_count >= 8:
                    best_shift_x     = shift_x
                    best_shift_y     = shift_y
                    best_confidence  = adjusted_confidence
                    best_match_count = len(good_matches)
                    # Calculate magnitude statistics
                    shift_magnitudes = np.sqrt(np.sum(shifts**2, axis=1))
                    mean_magnitude   = np.mean(shift_magnitudes)
                    std_magnitude    = np.std(shift_magnitudes)
                    best_stats       = (adjusted_confidence, mean_magnitude, std_magnitude)
                    
                    if verbose:
                        print(f"Method {method}: shift=({shift_x:.2f}, {shift_y:.2f}), confidence={adjusted_confidence:.2f}, matches={len(good_matches)}")
                
                # Visualize matches (for debugging)
                if verbose and visualize:  # Set to True to enable visualization
                    match_img = cv2.drawMatches(ref_clahe, kp1, target_clahe, kp2, good_matches, None)
                    cv2.imwrite(f"matches_{method}.jpg", match_img)
                    
            except Exception as e:
                if verbose:
                    print(f"Error with {method} matching: {str(e)}")
                continue
        
        # If no method found good matches
        if best_confidence == 0:
            if verbose:
                print("No reliable shift found with any feature matching method")
            return 0, 0, None
        
        x_shift_m = best_shift_x * abs(target_img.GetGeoTransform()[1])
        y_shift_m = best_shift_y * abs(target_img.GetGeoTransform()[5])

        if verbose:
            print(f"Final shift in meters: x={x_shift_m:.4f}m, y={y_shift_m:.4f}m")
            print(f"Confidence: {best_confidence:.4f}, Matches: {best_match_count}")
        
        return x_shift_m, y_shift_m, best_stats