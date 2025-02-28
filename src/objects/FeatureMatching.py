import os
import cv2
import fiona
import rasterio
import numpy                as np
import tools.visualizations as vis
import json

from rasterio.mask    import mask
from tqdm             import tqdm
from osgeo            import gdal, ogr, osr
from pathlib          import Path
from typing           import Dict
from scipy.signal     import correlate2d, fftconvolve
from scipy.fft        import fft2, ifft2

from scipy.optimize import minimize
from scipy.signal import correlate2d

from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union

class FeatureMatching:
    def __init__(self, config):
        self.config    = config
        self.images    = {} # Image dictionary containing all the image information
        self.offsets   = None

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
        shifts         = {} # TODO add to class attribute
        processed_set  = set() # TODO add to class attribute
        prev_shifts    = np.array([0, 0]) 
    
        # Convert dictionary values to a list for sorting purposes
        # Sort the images from NW to SE
        image_list = list(self.images.values())
#        image_list = self._sort_images_NW_to_SE()

        # Visualize the georeferenced images (for debugging)
#        vis.plot_georeferenced_images(image_list, first_idx=0, last_idx=2, title='Georeferenced Mugin Images', cmap='terrain', figsize=(10, 8), show_overlap=True)

        # Look through all images 
        for i, img in enumerate(image_list):
            if i == 0:
                # If it is the first image, add it to the processed set and continue
                processed_set.add(i)
                # Save the image without any changes
                img_name = os.path.basename(img["filepath"])
                img_name = img_name.replace(".tif", "_ortho.tif")
                self._save_image(img_name, gdal_img=img["gdalImg"])

                # Shift for the first image is zero
                shifts[img["filepath"]] = np.array([0, 0])
                continue

            overlap_shape, overlap_sr = self._get_overlap_shape(image_list[i-1]["filepath"],    # Base image (n-1)
                                                                image_list[i]["filepath"],      # Current image (n)
                                                                buffer_size=5,                  # Buffer around the overlap in meters
                                                                save_output_to_file=False)      # Save the overlap as a shapefile
            overlap_n_1 = self._cut_image(image_list[i-1],    # Base image (n-1)
                                           overlap_shape,
                                           overlap_sr, 
                                           save_overlap=True,
                                           save_path="C:\\DocumentsLocal\\07_Code\\SeaBee\\SeaBee_georef_seagulls\\DATA\\overlap")
            overlap_n_1 = self._cut_image(image_list[i-1],
                                          overlap_shape,
                                          overlap_sr)    # Base image (n-1)
            overlap_n = self._cut_image(image_list[i],        # Current image (n)
                                        overlap_shape,
                                        overlap_sr)      # Current image (n)
            # Extract the overlapping region between the current image and the previous image (n-1)
            #  If there is no overlap between the images, the function returns None, None
#            overlap_n_1, overlap_n = self._get_overlap_resampled(image_list[i-1]["filepath"],  # Base image (n-1)
#                                                                 image_list[i]["filepath"],    # Current image (n)
#                                                                 save_overlap=True,
#                                                                 save_path="C:\\DocumentsLocal\\07_Code\\SeaBee\\SeaBee_georef_seagulls\\DATA\\overlap",
#                                                                 no_data_value=-1)

            if (overlap_n_1 is None) or (overlap_n is None):
                # If there is no overlap (e.g. new row), save the image without any changes
                img_name = os.path.basename(img["filepath"])
                img_name = img_name.replace(".tif", "_ortho.tif")
                self._save_image(img_name, gdal_img=img["gdalImg"])

                # Set the shift for the image to zero
                shifts[img["filepath"]] = np.array([0, 0])
                processed_set.add(i)

                # Reset the tracking of the previous shifts
                prev_shifts = np.array([0, 0])
                continue

            # Get the image data of the image i
            #x_shift, y_shift = self._calculate_shift_manual(overlap_n_1, overlap_n, band=1, geotransform=image_list[i]["gdalImg"].GetGeoTransform())
            #x_shift, y_shift = self._2D_cross_correlation(overlap_n_1, overlap_n, band=0, max_shift_meter=10, geotransform_target=image_list[i]["gdalImg"].GetGeoTransform())
            x_shift, y_shift = self._shift_opticalFlow(overlap_n_1,
                                                       overlap_n,
                                                       geotransform_target=image_list[i]["gdalImg"].GetGeoTransform())
            
            # Store the shift for the image
            shifts[img["filepath"]] = np.array([x_shift, y_shift])
            # Calculate the total shift
            total_shift = prev_shifts + np.array([x_shift, y_shift])
            prev_shifts = total_shift

            # Apply the shift to the image
            img_name = image_list[i]["filepath"]
            self._apply_shift(self.images[img_name], total_shift[0], total_shift[1])
            processed_set.add(i)

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
            print("No overlap between images")
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
    
    def _crop_image(self, img_dict, overlap_shape, overlap_sr, save_overlap=False, save_path=None):
        """
        Crop an image to the bounds of a given shapefile
        """
        img_sr = osr.SpatialReference()
        img_sr.ImportFromWkt(img_dict["gdalImg"].GetProjection())
        if not img_sr.IsSame(overlap_sr):
            print("Warning: Image and overlap have different spatial references")
            # Transform the overlap to the image's spatial reference
            transform = osr.CoordinateTransformation(overlap_sr, img_sr)
            overlap_shape.Transform(transform)

        # Get the envelope of the overlap shape
        minX, maxX, minY, maxY = overlap_shape.GetEnvelope()
        print(f"Overlap envelope: minX={minX}, maxX={maxX}, minY={minY}, maxY={maxY}")

        # Get the geotransform of the image
        gt = img_dict["gdalImg"].GetGeoTransform()
        print(f"Image geotransform: {gt}")

        # Get the raster dimensions
        img_width  = img_dict["gdalImg"].RasterXSize
        img_height = img_dict["gdalImg"].RasterYSize
        print(f"Image dimensions: width={img_width}, height={img_height}")

        # Calculate the offsets for the window [pixel coordinates]
        # For most geotiffs, element 5 of geotransform is negative
        pixel_width  = abs(gt[1])
        pixel_height = abs(gt[5])
        print(f"Pixel dimensions: width={pixel_width}, height={pixel_height}")        

        # Convert envelope to pixel coordinates (origin of the image)
        x_origin = gt[0]
        y_origin = gt[3]

        # Calculate pixel coordinates
        # Calculate image extents in world coordinates 
        x_min_img = x_origin
        x_max_img = x_origin + img_width * gt[1]
        #x_min = int((minX - x_origin) / pixel_width)
        #x_max = int((maxX - x_origin) / pixel_width)
        #print(f"Pre-calculation: minY={minY}, maxY={maxY}, y_origin={y_origin}, pixel_height={pixel_height}")
        
#        # Y coordinates depend on whether origin is at top or bottom
#        if gt[5] < 0:  # Origin at top (normal case)
#            y_min = int((y_origin - maxY) / pixel_height)
#            y_max = int((y_origin - minY) / pixel_height)
#            print("Using 'Origin at top' calculation")
#        else:  # Origin at bottom
#            y_min = int((minY - y_origin) / pixel_height)
#            y_max = int((maxY - y_origin) / pixel_height)
#            print("Using 'Origin at bottom' calculation")

        if gt[5] > 0:  # Origin at bottom
            y_min_img = y_origin
            y_max_img = y_origin + img_height * gt[5]
        else:  # Origin at top
            y_max_img = y_origin
            y_min_img = y_origin + img_height * gt[5]
        print(f"Image world coordinates: x_min={x_min_img}, x_max={x_max_img}, y_min={y_min_img}, y_max={y_max_img}")

        if gt[5] > 0:  # Origin at bottom
            # For origin at bottom, y increases going up from the origin
            # Calculate y pixel coordinates relative to the origin (which is at the bottom left)
            y_min_px = int((minY - y_origin) / gt[5])
            y_max_px = int((maxY - y_origin) / gt[5])
            
            # The key fix: ensure y_min_px doesn't go below the image extent
            # For geotifs with origin at bottom, y_min_px should be >= 0
            # This is crucial for preserving the southern boundary
            if y_min_px < 0:
                print(f"Warning: Southern boundary extends below image bounds, adjusting y_min_px from {y_min_px} to 0")
                y_min_px = 0
        else:  # Origin at top
            # For origin at top, y increases going down from the origin
            y_min_px = int((y_origin - maxY) / -gt[5])
            y_max_px = int((y_origin - minY) / -gt[5])

        # X calculation is simpler and the same in both cases
        x_min_px = int((minX - x_origin) / gt[1])
        x_max_px = int((maxX - x_origin) / gt[1])

        print(f"Calculated pixel coords: x_min_px={x_min_px}, x_max_px={x_max_px}, y_min_px={y_min_px}, y_max_px={y_max_px}")

        # Ensure correct order (in case of negative pixel resolution)
        x_min_px, x_max_px = min(x_min_px, x_max_px), max(x_min_px, x_max_px)
        y_min_px, y_max_px = min(y_min_px, y_max_px), max(y_min_px, y_max_px)

        # Clip to image boundaries
        crop_x_min = max(0, x_min_px)
        crop_x_max = min(img_width, x_max_px)
        crop_y_min = max(0, y_min_px)
        crop_y_max = min(img_height, y_max_px)

        print(f"Final crop coordinates: crop_x_min={crop_x_min}, crop_x_max={crop_x_max}, crop_y_min={crop_y_min}, crop_y_max={crop_y_max}")

        # Calculate window dimensions
        crop_width  = crop_x_max - crop_x_min
        crop_height = crop_y_max - crop_y_min
        print(f"Crop dimensions: width={crop_width}, height={crop_height}")

        # Check if we have a valid window
        if crop_width  <= 0 or crop_height <= 0:
            print(f"Invalid window dimensions for {img_dict['filepath']}")
            return None

        # Create output dataset
        driver  = gdal.GetDriverByName('MEM')
        out_img = driver.Create('', crop_width, crop_height, img_dict["gdalImg"].RasterCount,
                                img_dict["gdalImg"].GetRasterBand(1).DataType)

        # Calculate new geotransform
        new_x_origin = x_origin + crop_x_min * gt[1]  # Use original gt[1], not pixel_width
        new_y_origin = y_origin + crop_y_min * gt[5]  # Preserve sign
        print(f"New geotransform origin: x={new_x_origin}, y={new_y_origin}")

        new_geotransform = (
            new_x_origin,
            gt[1],
            gt[2],
            new_y_origin,
            gt[4],
            gt[5]
        )
        print(f"New geotransform: {new_geotransform}")

        # Set projection and geotransform
        out_img.SetProjection(img_dict["gdalImg"].GetProjection())
        out_img.SetGeoTransform(new_geotransform)

        # Read data and write to output
        for i in range(1, img_dict["gdalImg"].RasterCount + 1):
            data = img_dict["gdalImg"].GetRasterBand(i).ReadAsArray(crop_x_min, crop_y_min, crop_width, crop_height)
            out_img.GetRasterBand(i).WriteArray(data)
        
        # Save if requested
        if save_overlap and save_path:
            # Save the cropped image to a new geotiff
            base_name = os.path.splitext(os.path.basename(img_dict["filepath"]))[0]
            save_name = os.path.join(save_path, f"{base_name}_crop.tif")
            self._save_image(save_name, gdal_img=out_img)

        #    # Save the overlap shape to a shapefile
        #    overlap_shape = overlap_shape.Clone()
        #    # Save the overlap shape to a shapefile as _crop.shp
        #    save_name = os.path.join(save_path, f"{base_name}_crop.shp")
        #    # save shapefile using fiona
        #    geojson = json.loads(overlap_shape.ExportToJson())
        #
        #    ##DEBUGING: Save to shapefile
        #    schema = {
        #        'geometry': 'Polygon',
        #        'properties': {'id': 'int'}
        #    }
        #    with fiona.open(save_name, 'w', 'ESRI Shapefile', schema) as c:
        #        c.write({
        #            'geometry': geojson,
        #            'properties': {'id': 1},
        #        })
        #
        #    print(f"Saved cropped image to {save_name}")

        return out_img
    
    def _cut_image(self, img_dict, overlap_shape, overlap_sr, save_overlap=False, save_path=None, no_data_value=None):
        """
        Crop image using gdalwarp with cutline
        """
        # Create a temporary shapefile for the cutline
        temp_shapefile = "/vsimem/temp_cutline.json"
    
        # Clone and transform shape if needed
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

    def _sort_images_NW_to_SE(self):
        """
        Sort images from Northwest to Southeast in a grid pattern.
    
        Args:
            image_list: List of GDAL dataset objects
        
        Returns:
            sorted_list: List of GDAL dataset objects sorted from NW to SE in rows
        """
        # Extract coordinates for each image
        image_coords = []
        image_list   = list(self.images.values())
        for img in image_list:
            # Get geotransform (contains coordinates info)
            geotransform = img["gdalImg"].GetGeoTransform()

            # Calculate center coordinates of the image
            width  = img["gdalImg"].RasterXSize
            height = img["gdalImg"].RasterYSize

            # Calculate center coordinates
            center_x = geotransform[0] + width * geotransform[1] / 2
            center_y = geotransform[3] + height * geotransform[5] / 2

            image_coords.append({
                'image': img["gdalImg"],
                'image_path': img["filepath"],
                'center_x': center_x,
                'center_y': center_y
            })

        # Find the approximate row structure
        # First, sort by Y coordinate (North to South)
        sorted_by_y = sorted(image_coords, key=lambda x: x['center_y'], reverse=False)

        # Calculate average Y differences between consecutive images
        y_diffs = [abs(sorted_by_y[i]['center_y'] - sorted_by_y[i+1]['center_y']) 
                   for i in range(len(sorted_by_y)-1)]

        if not y_diffs:
            # If only one image, return original list
            return image_list

        # Use median of differences to determine row boundaries
        median_y_diff = np.median(y_diffs)
        row_threshold = median_y_diff * 0.5  # 50% of median difference

        # Group images into rows
        rows        = []
        current_row = [sorted_by_y[0]]

        for i in range(1, len(sorted_by_y)):
            y_diff = abs(sorted_by_y[i]['center_y'] - sorted_by_y[i-1]['center_y'])
            
            if y_diff > row_threshold:
                # New row
                rows.append(current_row)
                current_row = [sorted_by_y[i]]
            else:
                # Same row
                current_row.append(sorted_by_y[i])
        
        rows.append(current_row)  # Add last row
        
        # Sort each row from West to East
        for row in rows:
            row.sort(key=lambda x: x['center_x'])
        
        # Flatten the rows into a single list
        sorted_list = [img['image'] for row in rows for img in row]
        
        return sorted_list

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
    
    def _shift_opticalFlow(self, ref_img, target_img, band=0, geotransform_target=None, max_shift_meter=10):
        """
        Calculate local shift between two overlapping images to properly align them, using optical flow.
        """
        if band >= ref_img.shape[0] or band >= target_img.shape[0]:
            raise ValueError(f"Band index {band} out of range: ref_img has {ref_img.shape[0]} bands, target_img has {target_img.shape[0]} bands")

        # Extract data from the masked arrays, handling NaNs properly
        ref_data = ref_img[band, :, :].data
        target_data = target_img[band, :, :].data

        # Create masks for valid pixels
        ref_mask = ~np.isnan(ref_data)
        target_mask = ~np.isnan(target_data)
        valid_mask = ref_mask & target_mask

        # Skip calculation if there's not enough valid data
        valid_percentage = np.sum(valid_mask) / valid_mask.size
        if valid_percentage < 0.3:
            print(f"Warning: Not enough valid data in overlap ({valid_percentage:.1%}). Skipping optical flow.")
            return 0, 0

        # Make copies of data to avoid modifying originals
        ref_ds = np.copy(ref_data)
        target_ds = np.copy(target_data)

        # Replace NaNs with zeros (or nearest valid values)
        ref_ds[~ref_mask] = 0
        target_ds[~target_mask] = 0

        # Convert to correct format for OpenCV
        ref_ds = ref_ds.astype(np.float32)
        target_ds = target_ds.astype(np.float32)

        # Enhance contrast in the valid regions for better feature matching
        if np.any(ref_mask):
            # Calculate statistics only from valid data
            ref_valid = ref_ds[ref_mask]
            ref_min, ref_max = np.percentile(ref_valid, [2, 98])  # Use percentiles to avoid outliers
            if ref_max > ref_min:
                # Apply contrast enhancement
                ref_ds = np.clip((ref_ds - ref_min) / (ref_max - ref_min) * 255, 0, 255)

        if np.any(target_mask):
            # Calculate statistics only from valid data
            target_valid = target_ds[target_mask]
            target_min, target_max = np.percentile(target_valid, [2, 98])
            if target_max > target_min:
                # Apply contrast enhancement
                target_ds = np.clip((target_ds - target_min) / (target_max - target_min) * 255, 0, 255)

        # Apply Gaussian blur to reduce noise
        ref_ds = cv2.GaussianBlur(ref_ds, (3, 3), 0)
        target_ds = cv2.GaussianBlur(target_ds, (3, 3), 0)

        try:
            # Compute optical flow with adjusted parameters for better subpixel accuracy
            flow = cv2.calcOpticalFlowFarneback(
                ref_ds, target_ds, None,
                pyr_scale=0.5,      # Pyramid scale (smaller values preserve more detail)
                levels=5,           # Number of pyramid levels (increase for better accuracy)
                winsize=21,         # Averaging window size (larger for smoother flow field)
                iterations=10,      # Number of iterations at each pyramid level
                poly_n=7,           # Size of pixel neighborhood used for polynomial expansion
                poly_sigma=1.5,     # Standard deviation of smoothing used in polynomial expansion
                flags=0
            )

            # Extract flow components
            x_flow = flow[:, :, 0]
            y_flow = flow[:, :, 1]

            # Calculate flow statistics for confidence assessment
            flow_magnitude = np.sqrt(x_flow**2 + y_flow**2)
            mean_magnitude = np.mean(flow_magnitude[valid_mask])
            std_magnitude = np.std(flow_magnitude[valid_mask])

            print(f"Flow statistics - Mean: {mean_magnitude:.4f}, StdDev: {std_magnitude:.4f}")

            # Robust estimation of shift (use median instead of mean for better robustness)
            x_shift = np.median(x_flow[valid_mask])
            y_shift = np.median(y_flow[valid_mask])

            # Print the raw shifts in pixels
            print(f"Raw pixel shift (median): x={x_shift:.4f}, y={y_shift:.4f}")

            # More advanced confidence assessment
            confidence = 1.0
            if std_magnitude > mean_magnitude * 0.8:  # High variability in flow
                confidence *= 0.7
                print("Warning: High variability in flow field")

            if mean_magnitude < 0.5:  # Very small flow might be noise
                confidence *= 0.9
                print("Note: Small overall displacement detected")

            print(f"Shift confidence estimate: {confidence:.2f}")

            # Adjust threshold for subpixel shifts - only apply zero threshold if confidence is low
            if confidence < 0.5:
                if abs(x_shift) < 1.0:
                    x_shift = 0
                if abs(y_shift) < 1.0:
                    y_shift = 0

            # Convert the shift to meters
            if geotransform_target is None:
                raise ValueError("Target image geotransform is missing")
            x_shift_m = x_shift * abs(geotransform_target[1])
            y_shift_m = y_shift * abs(geotransform_target[5])

            print(f"Final shift in meters: x={x_shift_m:.4f}m, y={y_shift_m:.4f}m")
            return x_shift_m, y_shift_m

        except cv2.error as e:
            print(f"OpenCV error: {str(e)}")
            # Try to diagnose the issue
            print(f"Image shapes - ref: {ref_ds.shape}, target: {target_ds.shape}")
            print(f"Value ranges - ref: {np.min(ref_ds)}-{np.max(ref_ds)}, target: {np.min(target_ds)}-{np.max(target_ds)}")
            return 0, 0

    def _2D_cross_correlation(self, ref_img, target_img, no_pix_lines=5, band=None, max_shift_meter=10, geotransform_target=None):
        """
        Calculate shift between two images using 2D cross-correlation.
        """
        # Select band and convert to correct format
        if band is not None:
            band_idx = band
        else:
            band_idx = 0

        ## DEBUGGING => Plotting the reference and target images
        #import matplotlib.pyplot as plt
        #plt.figure(figsize=(10, 4))
        #plt.subplot(1, 2, 1)
        #plt.imshow(ref_img[band_idx, :, :].data, cmap='gray')
        #plt.title('Reference Image')
        #plt.subplot(1, 2, 2)
        #plt.imshow(target_img[band_idx, :, :].data, cmap='gray')
        #plt.title('Target Image')
        #plt.show()

        # Extract "no_pix_lines" pixel lines from the reference image (vertical and horizontal)
        ref_hor_lines = ref_img[band_idx, :no_pix_lines, :].filled(np.nan)
        ref_ver_lines = ref_img[band_idx, :, :no_pix_lines].filled(np.nan)
        tar_hor_lines = target_img[band_idx, :no_pix_lines, :].filled(np.nan)
        tar_ver_lines = target_img[band_idx, :, :no_pix_lines].filled(np.nan)

        # Normalize the lines to 0-1 range
        ref_hor_lines = self._normalize_image(ref_hor_lines)
        ref_ver_lines = self._normalize_image(ref_ver_lines)
        tar_hor_lines = self._normalize_image(tar_hor_lines)
        tar_ver_lines = self._normalize_image(tar_ver_lines)

    def _calculate_shift_manual(self, ref_overlap_img, target_overlap_img, band=None, max_shift_meter = 7, geotransform=None):
        """
        Calculate shift between two images using 1D pixel line and cross-correlation.

        Returns shifts in whole pixels.
        """
        # Cost-function based approach
        # 1. Compute image convolutions using a Gaussian or Sobel filter.
        # 2. Calculate NCC between the two images.
        # 3. Calculate the shift penalty
        # 4. Optimize the constraint function to minimize the cost function.
        # Cost function: + high peak value, - low peak value (minimize) ALSO cost on shift magnitude

        def compute_ncc(shift):
            """
            Compute normalized cross-correlation for given shift.
            Returns NCC value and number of valid pixels used.
            """
            x_shift, y_shift = shift
            h, w = ref_img.shape

            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            # Apply shift
            x_shifted = x + x_shift
            y_shifted = y + y_shift

            # Mask for valid coordinates
            # Create mask for valid coordinates (within image boundaries)
            valid = (x_shifted >= 0) & (x_shifted < w) & (y_shifted >= 0) & (y_shifted < h)
            
            # Calculate percentage of valid pixels
            valid_percentage = np.sum(valid) / valid.size
            min_valid_percentage = 0.3  # Require at least 30% overlap
            
            if valid_percentage < min_valid_percentage:
                print(f"Warning: Only {valid_percentage:.1%} of pixels would be valid after shift. Minimum required: {min_valid_percentage:.1%}")
                return -1.0, 0

            # Sample shifted pixels
            x_sample = x_shifted[valid].astype(int)
            y_sample = y_shifted[valid].astype(int)

            # Get valid samples from both images
            ref_valid = ref_img[y[valid], x[valid]]
            target_valid = target_img[y_sample, x_sample]

            # Remove any remaining NaNs
            valid_values = ~np.isnan(ref_valid) & ~np.isnan(target_valid)
            if not np.any(valid_values):
                return -1.0, 0

            ref_valid = ref_valid[valid_values]
            target_valid = target_valid[valid_values]

            # Require more points for reliable correlation
            if len(ref_valid) < min_valid_pixels:
                return -1.0, 0

            # Compute NCC
            ref_mean = np.mean(ref_valid)
            target_mean = np.mean(target_valid)

            # Check for low variance (might indicate water or uniform areas)
            ref_std = np.std(ref_valid)
            target_std = np.std(target_valid)
            
            if ref_std < variance_threshold or target_std < variance_threshold:
                return -1.0, 0

            ref_norm = (ref_valid - ref_mean) / ref_std
            target_norm = (target_valid - target_mean) / target_std

            ncc = np.mean(ref_norm * target_norm)

            return ncc, len(ref_valid)
        
        def grid_search(center_x, center_y, radius, step):
            """
            Perform grid search around a center point with given radius and step size.
            """
            # initialize NCC to negative infinity and best_shift to (center_x, center_y) => (0, 0)
            best_ncc          = float('-inf')
            best_shift        = (center_x, center_y)
            best_valid_pixels = 0

            x_range = range(center_x - radius, center_x + radius + 1, step)
            y_range = range(center_y - radius, center_y + radius + 1, step)

            for x_shift in x_range:
                for y_shift in y_range:
                    ncc, valid_pixels = compute_ncc((x_shift, y_shift))
                    if ncc > best_ncc and valid_pixels >= min_valid_pixels:
                        best_ncc          = ncc
                        best_shift        = (x_shift, y_shift)
                        best_valid_pixels = valid_pixels
                        print(f"New best shift: ({x_shift}, {y_shift}), NCC: {ncc:.4f}, Valid pixels: {valid_pixels}")

            return best_shift, best_ncc, best_valid_pixels

        # Input validation
        if ref_overlap_img.shape != target_overlap_img.shape:
            raise ValueError("Images must have the same shape")
        if band is not None:
            band_idx = band - 1
        else:
            band_idx = 0
        
        if geotransform is None:
            pixelsize = 0.1  # Default pixel size in meters
        else:
            pixelsize = abs(geotransform[1])

        ## DEBUGGING => Plotting the reference and target images
        #import matplotlib.pyplot as plt
        #plt_band = 1
        #plt.figure(figsize=(10, 4))
        #plt.subplot(1, 2, 1)
        #plt.imshow(ref_overlap_img[plt_band, :, :].data, cmap='gray')
        #plt.title('Reference Image')
        #plt.subplot(1, 2, 2)
        #plt.imshow(target_overlap_img[plt_band, :, :].data, cmap='gray')
        #plt.title('Target Image')
        #plt.show()
    
        # Select band and convert to correct format
        # Extract the band and handle masked values
        ref_img    = ref_overlap_img[band_idx, :, :].filled(np.nan).astype(np.float32)
        target_img = target_overlap_img[band_idx, :, :].filled(np.nan).astype(np.float32)
    
        # Normalize images to 0-1 range, handling NaN values
        ref_min, ref_max       = np.nanmin(ref_img), np.nanmax(ref_img)
        target_min, target_max = np.nanmin(target_img), np.nanmax(target_img)

        ref_img    = (ref_img - ref_min) / (ref_max - ref_min)
        target_img = (target_img - target_min) / (target_max - target_min)

        ## DEBUGGING => Plotting the normalized reference and target images
        #import matplotlib.pyplot as plt
        #plt.figure(figsize=(10, 4))
        #plt.subplot(1, 2, 1)
        #plt.imshow(ref_img, cmap='gray')
        #plt.title('Reference Image')
        #plt.subplot(1, 2, 2)
        #plt.imshow(target_img, cmap='gray')
        #plt.title('Target Image')
        #plt.show()

        ## DEBUGGING => Print initial image statistics
        #print(f"Reference image stats - min: {np.nanmin(ref_img):.4f}, max: {np.nanmax(ref_img):.4f}, mean: {np.nanmean(ref_img):.4f}")
        #print(f"Target image stats - min: {np.nanmin(target_img):.4f}, max: {np.nanmax(target_img):.4f}, mean: {np.nanmean(target_img):.4f}")

        # Grid search parameters # TODO move to function arguments
        #calculate max shift in pixels from the max_shift_meter in meters
        max_search_radius  = int(max_shift_meter / pixelsize)
        min_valid_pixels   = 100  # Minimum overlap required
        variance_threshold = 0.02  # Minimum variance required for reliable correlation
        ncc_threshold      = 0.5  # Minimum NCC required for reliable correlation

        # First pass: Coarse search (step size = 4)
        print("\nCoarse grid search...")
        best_shift, best_ncc, best_valid_pixels = grid_search(
            0, 0, 
            max_search_radius,
            8)
    
        # Second pass: Fine search around best match (step size = 1)
        print("\nRefined grid search...")
        medium_shift, medium_ncc, medium_valid_pixels = grid_search(
            best_shift[0], best_shift[1], 
            8,  # Search radius for medium refinement
            4   # Step size for medium refinement
        )
    
        print("\nFine grid search...")
        refined_shift, refined_ncc, refined_valid_pixels = grid_search(
            medium_shift[0], medium_shift[1], 
            4,  # Search radius for final refinement
            1   # Step size for final refinement
        )

        print(f"\nFinal best shift: {refined_shift}, NCC: {refined_ncc:.4f}, Valid pixels: {refined_valid_pixels}")
        
        # Check if correlation is strong enough to apply shift
        if refined_ncc < ncc_threshold:
            print(f"Warning: Correlation too weak ({refined_ncc:.4f} < {ncc_threshold}). Ignoring shift.")
            return (0, 0)

        # Check for reasonable shift magnitude
        max_reasonable_shift = max_search_radius * 0.8  # 80% of max search radius
        shift_magnitude = np.sqrt(refined_shift[0]**2 + refined_shift[1]**2)
    
        if shift_magnitude > max_reasonable_shift:
            print(f"Warning: Shift magnitude ({shift_magnitude:.1f}) exceeds reasonable limit ({max_reasonable_shift:.1f}). Ignoring shift.")
            return (0, 0)

        return refined_shift
