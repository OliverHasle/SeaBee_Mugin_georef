import os
import cv2
import ast
import time

import numpy                as np
import tools.visualizations as vis

from scipy            import ndimage
from tqdm             import tqdm
from osgeo            import gdal, ogr, osr
from pathlib          import Path
from shapely.geometry import Polygon, MultiPolygon
from skimage.metrics  import structural_similarity as ssim

class Orthorectification:
    def __init__(self, config)                                                                                          -> None:
        self.frame_width            = 5                               # Width of the frame around the overlap in meters
        # Feature matching methods to use in self._transform_featureBased(..), options are 'sift', 'orb', 'akaze', 'brisk', 'kaze', 'surf', 'fast' 
        self.featureMatchingMethods = ast.literal_eval(config['ORTHORECTIFICATION']['FeatureMatching_methods'])
        self.use_optimization       = config["ORTHORECTIFICATION"]["use_iterative_optimization"].lower() == "true"
        self.band                   = int(config["ORTHORECTIFICATION"]["band_used_for_orthorectification"])

        self.config                 = config
        self.images                 = {}                  # Image dictionary containing all the image information
        self.processed_set          = set()               # Set of processed images
        self._load_geotiffs()
        self._clear_output_folder()

        # Convert dictionary values to a list for sorting purposes
        self.image_list             = list(self.images.values())                                   # List of images sorted by name
        # Generate lists with "None" values and the same length as image_list
        self.image_list_processed   = [None] * len(self.image_list)                                # List of images that have been processed
        
        self.rel_offset             = [None] * len(self.image_list)                                # List of relative offsets of the images (x, y) relative to its neighbors,                same order as image_list
        self.total_offset           = [None] * len(self.image_list)                                # List of total offsets of the images (x, y) relative to original position,               same order as image_list
        self.rel_rotations          = [None] * len(self.image_list)                                # List of relative rotations of the images (degree), relative to its neighbours           same order as image_list
        self.total_rotations        = [None] * len(self.image_list)                                # List of total rotations of the images (degree),                                         same order as image_list
        self.skews                  = [None] * len(self.image_list)                                # List of skew parameters of the images (x_skew, y_skew),                                 same order as image_list
        self.scaling                = [None] * len(self.image_list)                                # List of scaling factors of the images,                                                  same order as image_list
        self.confidence             = [None] * len(self.image_list)                                # List of confidence values for the transformations (0-1),                                same order as image_list
        self.mean_mag               = [None] * len(self.image_list)                                # List of mean magnitudes of the transformations,                                         same order as image_list
        self.std_div                = [None] * len(self.image_list)                                # List of standard deviations of the transformations,                                     same order as image_list
        self.matching_method        = [None] * len(self.image_list)                                # List of matching methods used to calculate the transformations for each image,          same order as image_list
        self.neighbours             = {}                                                           # Dict of overlapping images, key() = image idx, value() = "list of overlapping img idx", same order/index as image_list  

    def _load_geotiffs(self)                                                                                            -> None:
        """
        Loading all geotiff images in the image directory specified in the configuration file.
        """
        file_path = Path(self.config["MISSION"]["outputfolder"])
        print("Loading images from", file_path)

        try:
            tif_files   = list(file_path.glob("*.tif"))
            if not tif_files:
                raise FileNotFoundError("No geotiff files found in the specified directory.")

            for file in tqdm(tif_files, desc="Loading Geotiffs"):
                try:
                    # Create image information dirtionary for each image
                    # Load the image using gdal and store it in the dictionary
                    src = gdal.Open(str(file))
                    self.images[str(file)] = {
                        "filepath": str(file),
                        "imgName":  os.path.basename(file),
                        "gdalImg":  src
                    }
                except Exception as e:
                    print(f"Error loading {file}")
                    print(e)

            print("Loaded", len(self.images), "images.")
        except FileNotFoundError as e:
            print(e)
    def _clear_output_folder(self)                                                                                      -> None:
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
                    print(e)
                    raise(f"Error deleting {file_path}")
    def main_orthorectify_images(self, verbose=False)                                                                          -> None:
        """
        Process images for orthorectification.
        Base image:
            - The image which acts as a reference for the other image(s).

        Target image:
            - The image which is to be orthorectified.
        """
        # Finding all images which overlap (self.neighbours), img = 0 is the first img in self.image_list img = 1 is the second img in self.image_list etc. 
        self._find_image_neighbours()

        # Visualize the georeferenced images (for debugging)
#        vis.plot_georeferenced_images(image_list, first_idx=0, last_idx=2, title='Georeferenced Mugin Images', cmap='terrain', figsize=(10, 8), show_overlap=True)

        # Track time for the orthorectification process
        start = time.time()
        # Look through all images
        print("---------------------------------------------------")
        for idx_tgt, img_tgt in enumerate(tqdm(self.image_list, desc="Orthorectify images", unit="image")):
            if idx_tgt == 0:
                # First image has no shift
                self._process_image_with_no_transform(img_tgt, idx_tgt, np.array([0, 0]))
                continue

            # Initialize translations, rotations and skew parameters
            translations_tgt = []
            rotations_tgt    = []
            skew_params_tgt  = []
            stats_tgt        = []
            match_found      = False

            for idx_base, img_base in enumerate(self.image_list): # Loop through all images
                # Check if the images are overlapping and if the base image was already processed
                if ((not (idx_base == idx_tgt))            and    # Base image is not the same as the target image
                    (idx_base in self.neighbours[idx_tgt]) and    # Images are overlapping
                    (idx_tgt not in self.processed_set)    and    # Target image is not yet processed (this should always be the case)
                    (idx_base in self.processed_set)):            # Base image was processed already
                    # Images are overlapping and the base image was already processed
                    print(f" Estemating image {img_tgt['imgName']} position relative to image {img_base['imgName']}")

                    match_found = True

                    # Get the overlap shape and spatial reference
                    overlap_shape, overlap_sr = self._get_overlap_shape(img_tgt["filepath"],               # Base image (not getting changed)
                                                                        img_base["filepath"],                # Target image (changes applied to this image)
                                                                        buffer_size=self.frame_width)       # Buffer around the overlap in [meters]

                    if overlap_shape is None:
                        # If there is no overlap continue to the next image
                        print(f"No overlap between {img_tgt['imgName']} and {img_base['imgName']}")
                        continue

                    # Cut the images to the overlap region
                    overlap_base   = self._cut_image(img_tgt, overlap_shape, overlap_sr)
                    overlap_target = self._cut_image(img_base, overlap_shape, overlap_sr)

                    # Calculate the shift between the images
                    x_shift, y_shift, rotation, skew_params, scale_facors, stat, method = self._transform_hybrid(img_base["gdalImg"],
                                                                                                         img_tgt["gdalImg"],
                                                                                                         overlap_base,
                                                                                                         overlap_target,
                                                                                                         verbose)

                    # Add the shift to the list
                    translations_tgt.append(np.array([x_shift, y_shift]))
                    rotations_tgt.append(rotation)
                    skew_params_tgt.append(skew_params)
                    stats_tgt.append(stat)
                else:
                    continue

            if not match_found:
                # If no match was found, process the image with no transformation
                # The image is not overlapping with any other processed images
                self._process_image_with_no_transform(img_tgt, idx_tgt, np.array([0, 0]))
                continue

            # Calculate the average shift for the image; If there are no shifts, the average shift is zero
            avg_rel_trans = np.mean(translations_tgt, axis=0) if translations_tgt else np.array([0, 0])
            avg_rotation  = np.mean(rotations_tgt)            if rotations_tgt    else 0
            avg_skew      = np.mean(skew_params_tgt, axis=0)  if skew_params_tgt  else (0, 0)
            avg_stat      = np.mean(stats_tgt, axis=0)        if stats_tgt        else None

#            vis._visualize_transformations(translations_, rotations_, skew_params_, stats_, title="Transformations", figsize=(10, 8))

            # Calculate the average shift of the processed neighbours
            tmp_neighbour_shifts = []
            tmp_neighbour_rot    = []
            tmp_neighbour_skew   = []
            for idx_neighbour in self.neighbours[idx_tgt]:
                if idx_neighbour in self.processed_set:
                    # Get the shift of the neighbours
                    tmp_neighbour_shifts.append(self.total_offset[idx_neighbour])
                    tmp_neighbour_rot.append(self.total_rotations[idx_neighbour])
                    tmp_neighbour_skew.append(self.skews[idx_neighbour])
            avg_neighbour_trans = np.mean(tmp_neighbour_shifts, axis=0) if tmp_neighbour_shifts else np.array([0, 0])
            avg_neighbour_rot   = np.mean(tmp_neighbour_rot)            if tmp_neighbour_rot    else 0
            avg_neighbour_skew  = np.mean(tmp_neighbour_skew, axis=0)   if tmp_neighbour_skew   else (0, 0) # NOTE: Not used for now

            # Calculate the total shift for the image
            total_trans = avg_neighbour_trans + avg_rel_trans
            total_rot   = avg_neighbour_rot   + avg_rotation

            #NOTE: the rotation and skew of the neighbours are not taken into account at this time!

            # Store the shift for the image
            self.rel_offset[idx_tgt]      = avg_rel_trans
            self.total_offset[idx_tgt]    = total_trans
            self.rel_rotations[idx_tgt]   = avg_rotation
            self.total_rotations[idx_tgt] = total_rot
            self.skews[idx_tgt]           = avg_skew
            self.scaling[idx_tgt]         = scale_facors
            self.confidence[idx_tgt]      = float(avg_stat[0])
            self.mean_mag[idx_tgt]        = float(avg_stat[1])
            self.std_div[idx_tgt]         = float(avg_stat[2])
            self.matching_method[idx_tgt] = method

            # Apply the shift to the image
            self._apply_transform(idx_tgt, 
                                  x_shift  = total_trans[0],
                                  y_shift  = total_trans[1],
                                  rotation = avg_rotation,
                                  skew     = avg_skew,
                                  scale    = scale_facors)

            # Add the image to the processed set
            self.processed_set.add(idx_tgt)

        print("Processing done.")
        print("---------------------------------------------------")
        stop = time.time()
        print(f"Orthorectification took {stop - start:.2f} seconds.")
        #Show all the offsets in a graph
        vis.plot_offsets(self.offsets, self.confidence, self.mean_mag, self.std_div, title="Shifts between images", figsize=(10, 8))

    ##############################################
    ##  HELPER METHODS FOR ORTHORECTIFICATION   ##
    ##############################################

    def _process_image_with_no_transform(self, img, index, offset_neighbours)                                           -> np.ndarray:
        """
        Process the first image which doesn't need transformation.
        """
        # Save the image without any changes
        img_name = os.path.basename(img["filepath"])
        img_name = img_name.replace(".tif", "_ortho.tif")
        self._save_image(img_name, gdal_img=img["gdalImg"])

        self.rel_offset[index]      = np.array([0, 0])         # Relative offset is zero for an image with no transformation
        self.total_offset[index]    = offset_neighbours        # Total offset is the offset of the neighbours
        self.rel_rotations[index]   = 0                        # Rotation for an image with no transformation is zero
        self.total_rotations[index] = 0                        # Rotation for an image with no transformation is zero
        self.skews[index]           = np.array([0, 0])         # Skew for an image with no transformation is zero
        self.scaling[index]         = (1.0, 1.0)               # Scaling for an image with no transformation is (1.0, 1.0)
        self.confidence[index]      = 1                        # Confidence for an image with no transformation is 1
        self.mean_mag[index]        = 0                        # Mean magnitude for an image with no transformation is zero
        self.std_div[index]         = 0                        # Standard deviation for an image with no transformation is zero

        # Add the image to the processed set
        self.processed_set.add(index)

        # Load the image into the self.image_list_processed

        # Image path without name
        img_processed = {
            "imgName":  img_name,
            "gdalImg":  img["gdalImg"]
        }
        # Store into the self.image_list_processed in the right location
        # Find the index of the image in the self.image_list
        idx                            = self.image_list.index(img)
        self.image_list_processed[idx] = img_processed 

    def _apply_transform(self, img_idx, x_shift, y_shift, rotation, skew, scale=(1.0, 1.0))                             -> None:
        """
        Apply a full transformation (translation, rotation, and skew) to an image
        and save it to a new file.
    
        INPUT:
        - image: Image dictionary containing gdal dataset   (dict)
        - x_shift: X-shift in meters                        (float)
        - y_shift: Y-shift in meters                        (float)
        - rotation: Rotation angle in degrees               (float)
        - skew: Skew parameters (x_skew, y_skew) in degrees (tuple)
        - scale: Scale factors (x_scale, y_scale),          (tuple)       default is (1.0, 1.0)
        """
        # Get the gdal dataset for the image
        ds = self.image_list[img_idx]["gdalImg"]
    
        if isinstance(skew, np.ndarray):
            skew = tuple(skew.tolist())
        if isinstance(scale, np.ndarray):
            scale = tuple(scale.tolist())

        # If no transformation needed, just save the image
        if x_shift == 0 and y_shift == 0 and rotation == 0 and skew == (0, 0) and scale == (1.0, 1.0):
            img_name = os.path.basename(self.image_list[img_idx]["filepath"])
            img_name = img_name.replace(".tif", "_ortho.tif")
            self._save_image(img_name, gdal_img=ds)
            return
        
        # Create a copy of the dataset
        driver  = gdal.GetDriverByName("GTiff")
        ds_copy = driver.CreateCopy("/vsimem/temp.tif", ds, 0)
    
        # Get the geotransform
        old_gt    = ds.GetGeoTransform()
        ds_width  = ds.RasterXSize
        ds_height = ds.RasterYSize

        # Calculate new geotransform that incorporates translation, rotation, skew and scale
        new_gt = self._calculate_transformed_geotransform(old_gt, ds_width, ds_height, x_shift, y_shift, rotation, skew, scale)
        # TODO: Add calculate new geotransform with optimization function, considering confidence, mean magnitude and standard deviation (NCC)
        
        # Set the new geotransform
        ds_copy.SetGeoTransform(new_gt)
    
        # Get the image name
        img_name = os.path.basename(self.image_list[img_idx]["filepath"])
        img_name = img_name.replace(".tif", "_ortho.tif")
        
        # Save the transformed image
        self._save_image(img_name, gdal_img=ds_copy)
        gdal.Unlink("/vsimem/temp.tif")

    def _calculate_transformed_geotransform(self, geotransform, ds_width, ds_height, x_shift, y_shift, rotation_deg, skew, scale=(1.0, 1.0)) -> tuple:
        """
        Calculate a new geotransform that incorporates translation, rotation, and skew.
        
        INPUTS:
        - geotransform: Original geotransform               (6-element tuple)
        - x_shift: X-shift in meters                        (float)
        - y_shift: Y-shift in meters                        (float)
        - rotation_deg: Rotation angle in degrees           (float)
        - skew: Skew parameters (x_skew, y_skew) in degrees (tuple)
        - scale: Scale factors (x_scale, y_scale),          (tuple)       default is (1.0, 1.0)

        RETURNS:
        - New geotransform with applied transformations (6-element tuple)
        """
        # Extract original geotransform components
        x0, px_width, row_rot, y0, col_rot, px_height = geotransform
        
        # Convert degrees to radians
        rotation_rad = np.radians(rotation_deg)
        skew_x_rad   = np.radians(skew[0])
        skew_y_rad   = np.radians(skew[1])

        # Extract scale components
        scale_x, scale_y = scale

        # Calculate center of image in pixel coordinates
        center_pixel_x = ds_width / 2
        center_pixel_y = ds_height / 2

        # Convert center to world coordinates
        center_world_x = x0 + center_pixel_x * px_width + center_pixel_y * row_rot
        center_world_y = y0 + center_pixel_x * col_rot  + center_pixel_y * px_height
        
        # Initialize transformation matrices
        # Translation matrix
        T_to_origin = np.array([
            [1, 0, -center_world_x],
            [0, 1, -center_world_y],
            [0, 0, 1]
        ])

        T_from_origin = np.array([
            [1, 0, center_world_x],
            [0, 1, center_world_y],
            [0, 0, 1]
        ])

        # Translation matrix for the shift
        T_shift = np.array([
            [1, 0, x_shift],
            [0, 1, y_shift],
            [0, 0, 1]
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(rotation_rad), -np.sin(rotation_rad), 0],
            [np.sin(rotation_rad),  np.cos(rotation_rad), 0],
            [0,                     0,                    1]
        ])
        
        # Skew matrix (shear transform)
        S = np.array([
            [1,                  np.tan(skew_x_rad), 0],
            [np.tan(skew_y_rad), 1,                  0],
            [0,                  0,                  1]
        ])

        # Scale matrix
        SC = np.array([
            [scale_x, 0,       0],
            [0,       scale_y, 0],
            [0,       0,       1]
        ])
        
        # Original affine matrix from geotransform
        original = np.array([
            [px_width, row_rot,   x0],
            [col_rot,  px_height, y0],
            [0,        0,         1]
        ])
        
        # Combined transformation (apply in order: original → skew → rotate → translate)
        # The transformation order matters: we first apply the original transform,
        # then apply our additional transformations
#        combined = T @ R @ S @ SC @ original
        combined = T_from_origin @ T_shift @ S @ R @ SC @ T_to_origin @ original

        # Extract new geotransform elements
        new_geotransform = (
            combined[0, 2],  # x0
            combined[0, 0],  # px_width
            combined[0, 1],  # row_rot
            combined[1, 2],  # y0
            combined[1, 0],  # col_rot
            combined[1, 1]   # px_height
        )
        
        return new_geotransform

    def _get_overlap_shape(self, img_name1, img_name2, buffer_size=0, save_output_to_file=False)                        -> tuple:
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

    def _cut_image(self, img_dict, overlap_shape, overlap_sr, save_overlap=False, save_path=None, no_data_value=None)   -> gdal.Dataset:
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

    def _apply_shift(self, image, x_shift, y_shift)                                                                     -> None:
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

    def _save_image(self, img_name, gdal_img, extension=".tif")                                                         -> bool:
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

    def _find_image_neighbours(self)                                                                                    -> None:
        """
        Get the neighbours of all images in the image list, and store the indices of the neighbours in a dictionary.
        => Images are neighbours if they overlap.
        => The neighbours are stored as a dictionary with the image index as key and a list of neighbour indices as value.
        => The order of the images in the "self.neighbours" dictionary is the same as in the "self.image_list" list.
        """
        if not hasattr(self, 'neighbours'):
            # ROBUSTNESS: This should never happen
            self.neighbours = {}

        enum_img_list = list(enumerate(self.image_list))
        # Pre-compute bounds/corners
        image_bounds = {}
        print("---------------------------------------------------")
        for i, img in tqdm(enum_img_list, desc="Extract image corners", unit="image"):
            try:
                image_bounds[i] = self._get_image_corners(img["gdalImg"])
            except Exception as e:
                print(f"Error computing bounds/corners for image {i} ({img.get('imgName', 'unknown')}): {str(e)}")

        print("---------------------------------------------------")
        for i, img_stub in tqdm(enum_img_list, desc="Finding image neighbours", unit="image"):
            neighbours = []
            img_bounds = image_bounds.get(i)
        
            if img_bounds is None:
                print(f"Skipping image {i} because bounds could not be computed")
                continue

            for j, other_img_stub in enum_img_list:
                if i == j:
                    continue

                other_img_bounds = image_bounds.get(j)
                if other_img_bounds is None:
                    continue

                # Check if the images overlap
                overlap = self._check_overlap(img_bounds, other_img_bounds)

                if overlap:
                    neighbours.append(j)

            # Store the neighbours
            self.neighbours[i] = neighbours

    def _get_picture_type(self, reference_img, target_img, band=0, verbose=False)                                       -> str:
        """
        Determine the type of image based on:
        - Standard Deviation
        - Histogram
        - Edge Density
        - Similarity between images (SSIM or correlation)
        
        INPUT:
        - reference_img: Reference image (GDAL dataset)
        - target_img: Target image (GDAL dataset)
        
        OUTPUT:
        - picture_type: Type of picture (string) - 'featurebased', 'opticalflow', 'phasecorrelation', 'optimization'
        """
        # Set edge density threshold for target image (tuning parameter)
        tgt_edge_density_threshold = 60.0

        # Images must have the same dimensions
        if reference_img.RasterXSize != target_img.RasterXSize or reference_img.RasterYSize != target_img.RasterYSize:
            # Resample images to match
            # Read band data
            ref_band           = reference_img.GetRasterBand(band + 1)
            target_band        = target_img.GetRasterBand(band + 1)
            ref_data, tgt_data = self._synch_image_dimensions(ref_band.ReadAsArray(), target_band.ReadAsArray())
        else:
            ref_data = np.copy(reference_img.ReadAsArray())
            tgt_data = np.copy(target_img.ReadAsArray())

        # Calculate image statistics
#        ref_std    = np.std(ref_data)
#        target_std = np.std(tgt_data)
#        ref_mean   = np.mean(ref_data)
#        tgt_mean   = np.mean(tgt_data)

        # Calculate texture metrics (e.g., edge density)
        ref_edges        = ndimage.sobel(ref_data)
        tgt_edges        = ndimage.sobel(tgt_data)
        ref_edge_density = np.mean(np.abs(ref_edges))
        tgt_edge_density = np.mean(np.abs(tgt_edges))

        # Calculate similarity between images
        similarity_index = ssim(ref_data, tgt_data)

        # Water detection using histogram concentration
        ref_hist = np.histogram(ref_data, bins=10)[0]
        tgt_hist = np.histogram(tgt_data, bins=10)[0]

        # Calculate histogram concentration (percentage in most populated bins)
        ref_hist_sorted        = np.sort(ref_hist)[::-1]
        tgt_hist_sorted        = np.sort(tgt_hist)[::-1]
        ref_hist_concentration = (ref_hist_sorted[0] + ref_hist_sorted[1]) / np.sum(ref_hist)
        tgt_hist_concentration = (tgt_hist_sorted[0] + tgt_hist_sorted[1]) / np.sum(tgt_hist)
    
        # Determine if these are primarily water images
        primarily_water = ref_hist_concentration > 0.75 and tgt_edge_density  < tgt_edge_density_threshold

        has_land_features = False
        if len(ref_hist) > 3:
            # Check for peaks in the histogram that might indicate land features
            # Skip the first bin (often water) and look for other significant peaks
            for i in range(1, len(ref_hist)-1):
                if (ref_hist[i] > ref_hist[i-1] and ref_hist[i] > ref_hist[i+1] and
                    ref_hist[i] > 0.1 * np.max(ref_hist)):
                    has_land_features = True
                    break
        if verbose:
            print(f"Image Analysis Results:")
            print(f"  - Ref edge density: {ref_edge_density:.2f}")
            print(f"  - Target edge density: {tgt_edge_density:.2f}")
            print(f"  - Similarity index: {similarity_index:.2f}")
            print(f"  - Ref histogram concentration: {ref_hist_concentration:.2f}")
            print(f"  - Tgt histogram concentration: {tgt_hist_concentration:.2f}")
            print(f"  - Is primarily water region: {primarily_water}")
            print(f"  - Has land features: {has_land_features}")

        # Decision logic with better feature detection
        if ref_edge_density > 60.0 and tgt_edge_density > tgt_edge_density_threshold:
            # Lower threshold for images with distinct features
            return 'featurebased'
        elif has_land_features and not primarily_water:
            # If we detect land features in a mixed scene, use feature-based
            return 'featurebased'
        elif similarity_index > 0.7 and not primarily_water:
            # Very similar images with gradual changes
            return 'opticalflow'
        elif primarily_water and not has_land_features:
        # Pure water regions work well with phase correlation
            return 'phasecorrelation'
        else:
            # Default fallback for other cases
            return 'phasecorrelation'

    def _prepare_images_for_processing(self, img1, img2, b, verbose=False)                                              -> tuple:
        """
        Prepare images for processing:
        
        INPUT:
        - img1: First image (GDAL dataset)
        - img2: Second image (GDAL dataset)

        OUTPUT:
        """
        if img1 is None or img2 is None:
            return None, None
        # Read band data (gdal bands are 1-based)
        bands1   = img1.GetRasterBand(b + 1)
        bands2   = img2.GetRasterBand(b + 1)
        data1    = bands1.ReadAsArray()
        data2    = bands2.ReadAsArray()
        noData1  = bands1.GetNoDataValue()
        noData2  = bands2.GetNoDataValue()

        # Ensure same dimensions
        if data1.shape != data2.shape:
            if verbose:
                print(f"Images have different dimensions. Resampling to match.")
            ds1, ds2 = self._synch_image_dimensions(data1, data2)
        else:
            ds1 = np.copy(data1)
            ds2 = np.copy(data2)
        return ds1, ds2, noData1, noData2

    ###########################
    ##     STATIC METHODS    ##
    ###########################
    @staticmethod
    def _get_image_corners(gdal_img_dataSet)                                                                            -> list:
        """
        Get the bounds of an image.
        """
        gt = gdal_img_dataSet.GetGeoTransform()

        if gt is None:
            raise ValueError("Invalid geotransform: No bounds available")

        # Get the raster dimensions
        x_size = gdal_img_dataSet.RasterXSize
        y_size = gdal_img_dataSet.RasterYSize

        # Top left corner (0,0)
        x1 = gt[0]
        y1 = gt[3]

        # Top right corner (x_size,0)
        x2 = gt[0] + x_size * gt[1] + 0 * gt[2]
        y2 = gt[3] + x_size * gt[4] + 0 * gt[5]

        # Bottom right corner (x_size,y_size)
        x3 = gt[0] + x_size * gt[1] + y_size * gt[2]
        y3 = gt[3] + x_size * gt[4] + y_size * gt[5]

        # Bottom left corner (0,y_size)
        x4 = gt[0] + 0 * gt[1] + y_size * gt[2]
        y4 = gt[3] + 0 * gt[4] + y_size * gt[5]

        corners = [
            (float(x1), float(y1)),
            (float(x2), float(y2)),
            (float(x3), float(y3)),
            (float(x4), float(y4))
        ]

        return corners

    @staticmethod
    def _check_overlap(img_corners, other_img_corners, epsilon=1e-10)                                                   -> bool:
        """
        Check if two images overlap.
        """
        def get_axes(corners):
            """Get all potential separating axes for a polygon"""
            axes = []
            for i in range(len(corners)):
                # Get edge vector
                x1, y1 = corners[i]
                x2, y2 = corners[(i + 1) % len(corners)]
                edge = (x2 - x1, y2 - y1)
                
                # Get perpendicular vector (normal)
                normal = (-edge[1], edge[0])
                
                # Normalize the normal vector
                length = (normal[0]**2 + normal[1]**2)**0.5
                if length > epsilon:  # Avoid division by zero
                    normal = (normal[0]/length, normal[1]/length)
                    axes.append(normal)
            return axes
    
        def project_polygon(corners, axis):
            """Project a polygon onto an axis"""
            min_proj = float('inf')
            max_proj = float('-inf')
            
            for x, y in corners:
                projection = x * axis[0] + y * axis[1]
                min_proj = min(min_proj, projection)
                max_proj = max(max_proj, projection)
                
            return min_proj, max_proj
        
        # Get all potential separating axes
        all_axes = get_axes(img_corners) + get_axes(other_img_corners)
        
        # Check for separation along each axis
        for axis in all_axes:
            proj1 = project_polygon(img_corners, axis)
            proj2 = project_polygon(other_img_corners, axis)
            
            # If projections don't overlap, polygons don't overlap
            if proj1[1] < proj2[0] - epsilon or proj2[1] < proj1[0] - epsilon:
                return False
        
        # If we get here, no separating axis was found, so polygons overlap
        return True

    @staticmethod
    def _world_to_pixel(x, y, geotransform)                                                                             -> tuple:
        det     = geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4]
        if np.isclose(det, 0):
            raise ValueError("Invalid geotransform: Determinant is zero")

        pixel_x = (geotransform[5] * (x - geotransform[0]) - 
                  geotransform[2] * (y - geotransform[3])) / det
        pixel_y = (-geotransform[4] * (x - geotransform[0]) + 
                  geotransform[1] * (y - geotransform[3])) / det
        return np.round(pixel_x).astype(int), np.round(pixel_y).astype(int)

    @staticmethod
    def _save_overlap_geotiff(geo_array, output_path)                                                                   -> bool:
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

    @staticmethod
    def _synch_image_dimensions(img1, img2)                                                                             -> tuple:
        """
        This function takes two images which have different dimentions,
        """
        # Check if image is gray or color
        common_height = min(img1.shape[0], img2.shape[0])
        common_width  = min(img1.shape[1], img2.shape[1])

        # Use INTER_AREA for downsampling and INTER_CUBIC for upsampling
        if common_width < img1.shape[1] or common_height < img1.shape[0]:
            img1_interp = cv2.INTER_AREA
        else:
            img1_interp = cv2.INTER_CUBIC

        if common_width < img2.shape[1] or common_height < img2.shape[0]:
            img2_interp = cv2.INTER_AREA
        else:
            img2_interp = cv2.INTER_CUBIC
                
        # Resize both images to the common size
        ref_ds    = cv2.resize(img1, (common_width, common_height), interpolation=img1_interp)
        target_ds = cv2.resize(img2, (common_width, common_height), interpolation=img2_interp)
        return ref_ds, target_ds
    
    def _transform_hybrid(self, ref_img, target_img, ref_overlap, target_overlap, verbose=False)                -> tuple:
        """
        Hybrid orthorectification method that combines feature-based matching, optical flow, and phase correlation.
        The method uses the full images for feature-based matching and the overlap region for optical flow and phase correlation.
        The final transformation is a weighted average of the results from each method.

        Returns:
            tuple: (x_shift, y_shift, rotation_angle, skew_params, confidence_stats)
            Where:
            - x_shift, y_shift: Translation in meters
            - rotation_angle: Rotation in degrees
            - skew_params: Tuple containing skew parameters (x_skew, y_skew)
            - confidence_stats: Tuple with (confidence, mean_magnitude, std_deviation)
        """
        ## DEUBGGING
        #import matplotlib.pyplot as plt
        #plt.imshow(target_img.GetRasterBand(2).ReadAsArray())
        
        if ref_img is None or target_img is None:
            if verbose:
                print("Error: One of the input images is None")
            return 0, 0, 0, (0, 0), (1.0, 1.0), None, None
        
        #################################################################################################################
        # Step 1: Figure out which kind of picture(s) we have and the method to use (based on the overlap region)
        #################################################################################################################
        picture_type = self._get_picture_type(ref_overlap, target_overlap, band=self.band, verbose=verbose)

        if verbose:
            print(f"Selected method based on image type: {picture_type}")

        x_shift, y_shift, rotation, skew_params, scale, stats, method = 0, 0, 0, (0, 0), (1.0, 1.0), None, None

        try:
            #################################################################################################################
            # Step 2a: Try feature-based matching with full transformation detection (use whole image)
            #################################################################################################################
            if picture_type == 'featurebased':
                x_shift, y_shift, rotation, skew_params, scale, stats = self._transform_featureBased(ref_img, 
                                                                                              target_img,
                                                                                              band      = self.band,
                                                                                              verbose   = verbose,
                                                                                              visualize = False)

            #################################################################################################################
            # Step 2b: Try optical flow with multiscale approach (use overlap region only)
            #################################################################################################################
            elif picture_type == 'opticalflow':
                x_shift, y_shift, rotation, skew_params, scale, stats = self._shift_opticalFlow_multiscale(ref_overlap,
                                                                                                    target_overlap,
                                                                                                    band        = self.band,
                                                                                                    verbose     = verbose)

            #################################################################################################################
            # Step 2c: Try phase correlation (only detects translation) with bandpass filtering (use overlap region only)
            #################################################################################################################
            elif picture_type == 'phasecorrelation':
                x_shift, y_shift, stats = self._shift_phaseCorrelation(ref_overlap,
                                                                       target_overlap,
                                                                       band        = self.band,
                                                                       verbose     = verbose)
                rotation     = 0
                skew_params  = (0, 0)
                scale        = (1.0, 1.0)
            method = picture_type

        except Exception as e:
            if verbose:
                print(f"Error applying {picture_type} method: {str(e)}")

        #################################################################################################################
        # Step 3: Check if we have valid results
        #################################################################################################################
        if (stats is None or stats[0] < 0.3) and (self.use_optimization is True):
            if verbose:
                print(f"Low confidence with {picture_type} method: {stats[0] if stats else 'None'}")
                print("Trying alternative methods...")

            # Try other methods and use the one with highest confidence
            alternative_results = []

            # Only try methods we haven't already tried
            if picture_type != 'featurebased' and picture_type != 'phasecorrelation': # Don't try feature-based if the picture type is phase correlation (mostly water regions, unlikely to have features)
                try:
                    alt_x, alt_y, alt_rot, alt_skew, alt_scale, alt_stats = self._transform_featureBased(
                        ref_img, target_img, band=self.band, verbose=verbose
                    )
                    if alt_stats is not None and alt_stats[0] >= 0.3:
                        alternative_results.append((alt_x, alt_y, alt_rot, alt_skew, alt_scale, alt_stats, 'featurebased'))
                except Exception:
                    pass

            if picture_type != 'opticalflow':
                try:
                    alt_x, alt_y, alt_rot, alt_skew, alt_scale, alt_stats = self._shift_opticalFlow_multiscale(
                        ref_overlap, target_overlap, band=self.band, verbose=verbose
                    )
                    if alt_stats is not None and alt_stats[0] >= 0.3:
                        alternative_results.append((alt_x, alt_y, alt_rot, alt_skew, alt_scale, alt_stats, 'opticalflow'))
                except Exception:
                    pass

            if picture_type != 'phasecorrelation':
                try:
                    alt_x, alt_y, alt_stats = self._shift_phaseCorrelation(
                        ref_overlap, target_overlap, band=self.band, verbose=verbose
                    )
                    if alt_stats is not None and alt_stats[0] >= 0.3:
                        alternative_results.append((alt_x, alt_y, 0, (0, 0), (1.0, 1.0), alt_stats, 'phasecorrelation'))
                except Exception:
                    pass

            # If we found any good alternatives, use the one with highest confidence
            if alternative_results:
                # Sort by confidence (descending)
                alternative_results.sort(key=lambda x: x[5][0], reverse=True)
                best_alt = alternative_results[0]
                
                # Check if the best alternative is better than the original method
                if (((stats == None) and (best_alt is not None)) or 
                    ((stats[0] is not None) and (best_alt[5][0] > stats[0]))):
                    x_shift, y_shift, rotation, skew_params, scale, stats, method = best_alt
                if best_alt[5][0] > stats[0]:
                    x_shift, y_shift, rotation, skew_params, scale, stats, method = best_alt
#                    x_shift, y_shift, rotation, skew_params, scale_factors, stats = best_alt[0], best_alt[1], best_alt[2], best_alt[3], best_alt[4], best_alt[5]
#                    method                                                        = best_alt[6]
                else:
                    pass

                if verbose:
                    print(f"Using alternative method {best_alt[6]} with confidence {stats[0]:.2f}")

        #################################################################################################################
        # Step 4: If we still don't have a good result, return zeros with low confidence
        #################################################################################################################
        if stats is None:
            # x_shift, y_shift, rotation, skew_params, scale, stats,
            if verbose:
                print("No valid transformation found")
            return 0, 0, 0, (0, 0), (1.0, 1.0), (0.1, 0, 0), None

        #################################################################################################################
        # Step 5: Consider optimization refinement for moderate confidence results
        #################################################################################################################
        # TODO: To be implemented

        if self.config["ORTHORECTIFICATION"]["only_transl_rotation"] is True:
            # Only return translation and rotation, set skew and scale to zero
            skew_params = (0, 0)
            scale       = (1.0, 1.0)

        # Check if the transformation is too large
        if ((np.sqrt(x_shift**2 + y_shift**2) > float(self.config["ORTHORECTIFICATION"]["maximal_allowed_trans_meter"])) or
            (np.abs(rotation) > float(self.config["ORTHORECTIFICATION"]["maximal_allowed_rot_deg"]))                     or 
            (np.abs(skew_params[0]) > float(self.config["ORTHORECTIFICATION"]["maximal_allowed_skew_deg"]))              or
            (np.abs(skew_params[1]) > float(self.config["ORTHORECTIFICATION"]["maximal_allowed_skew_deg"]))              or
            (np.abs(scale[0] - 1.0) > float(self.config["ORTHORECTIFICATION"]["maximal_allowed_scale_factor"])) * 0.1    or
            (np.abs(scale[1] - 1.0) > float(self.config["ORTHORECTIFICATION"]["maximal_allowed_scale_factor"])) * 0.1):
            if verbose:
                print(f"Transformation too large: shift=({x_shift:.2f}, {y_shift:.2f}), " +
                      f"rotation={rotation:.2f}°, skew=({skew_params[0]:.2f}, {skew_params[1]:.2f})")
            return 0, 0, 0, (0, 0), (1.0, 1.0), (0.1, 0, 0), None
        # Return the results from the best method
        return x_shift, y_shift, rotation, skew_params, scale, stats, method

    def _transform_featureBased(self, ref_img, target_img, band=0, verbose=False, visualize=False)                      -> tuple:
        """
        Calculate transformation between two images using feature-based matching & homography.
        (Translation, Rotation, and Skew).

        INPUT:
            ref_img    (gdal.Dataset):  Reference image (image overlap + frame)
            target_img (gdal.Dataset):  Target image    (image overlap + frame)
            band       (int):           Band index to use (default: 0)
            verbose    (bool):          Print debug messages in this function (default: False)
            visualize  (bool):          Visualize matches and homography (default: False)

        OUTPUT:
            tuple: (x_shift, y_shift, rotation_angle, skew_params, confidence_stats)

        FURTHER INFORMATION:
            https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html
        """
        valid_data_required = 0.15 # 15% valid data required for feature-based matching
        good_ratio          = 0.75  # Ratio of good matches to consider (top 75% of matches are considered good, the rest is discarded)

        if ref_img is None or target_img is None:
            if verbose:
                print("Error: One of the input images is None")
            return 0, 0, 0, (0, 0), (1, 1), None

        # Read band data (gdal bands are 1-based)
        ref_band    = ref_img.GetRasterBand(band + 1)
        target_band = target_img.GetRasterBand(band + 1)

        # Robustness check for empty bands
        if ref_band is None or target_band is None:
            print("ERROR: wrong band index")
            return 0, 0, 0, (0, 0), (1, 1), None

        ref_data    = ref_band.ReadAsArray()
        target_data = target_band.ReadAsArray()

        if ref_data is None or target_data is None:
            return 0, 0, 0, (0, 0), (1, 1), None

        # Get NoData values
        ref_nodata    = ref_band.GetNoDataValue()
        target_nodata = target_band.GetNoDataValue()

        # Get dimensions of both images
        ref_height, ref_width       = ref_data.shape
        target_height, target_width = target_data.shape

        # Ensure same dimensions of the images (this should not happen with "whole images").
        if ref_data.shape != target_data.shape:
            if verbose:
                print(f"Images have different dimensions. Resampling to match.")

                # For whole image approach, we can use larger dimensions to preserve features
            max_height = max(ref_data.shape[0], target_data.shape[0])
            max_width  = max(ref_data.shape[1], target_data.shape[1])

            # Use appropriate interpolation method based on whether we're upsampling or downsampling
            ref_interp    = cv2.INTER_AREA if max_width < ref_width    or max_height < ref_height    else cv2.INTER_CUBIC
            target_interp = cv2.INTER_AREA if max_width < target_width or max_height < target_height else cv2.INTER_CUBIC

            ref_ds    = cv2.resize(ref_data,    (max_width, max_height), interpolation=ref_interp)
            target_ds = cv2.resize(target_data, (max_width, max_height), interpolation=target_interp)
        else:
            ref_ds    = np.copy(ref_data)
            target_ds = np.copy(target_data)

        ref_data, target_data, ref_nodata, target_nodata = self._prepare_images_for_processing(ref_img, target_img, band)

        # Create masks for valid pixels
        ref_mask    = np.ones_like(ref_ds,    dtype=bool)
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
#        # DEBUGING
#        import matplotlib.pyplot as plt
#        plt.imshow(valid_mask)

        # If using whole image, we're more tolerant of low valid percentages
        # Skip calculation if there's not enough valid data
        valid_percentage = np.sum(valid_mask) / valid_mask.size
        if valid_percentage < valid_data_required:
            if verbose:
                print(f"Warning: Not enough valid data in overlap ({valid_percentage:.1%}). Skipping matching.")

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
            """
            Normalize image for feature detection (uint8, contrast enhancement).
            """
            if np.any(mask):
                img_valid        = img[mask]
                min_val, max_val = np.percentile(img_valid, [2, 98])
                if max_val > min_val:
                    img_norm       = np.zeros_like(img)
                    img_norm[mask] = np.clip((img[mask] - min_val) / (max_val - min_val) * 255, 0, 255)
                    return img_norm.astype(np.uint8)
            return np.zeros_like(img, dtype=np.uint8)
    
        ref_uint8    = normalize_for_features(ref_ds,    ref_mask)
        target_uint8 = normalize_for_features(target_ds, target_mask)
    
        # Convolution with a sharpening kernel to enhance features
        kernel       = np.array([[-1, -1, -1], 
                                 [-1,  9, -1], 
                                 [-1, -1, -1]])
        ref_sharp    = cv2.filter2D(ref_uint8,    -1, kernel)
        target_sharp = cv2.filter2D(target_uint8, -1, kernel)

        ## DEBUGING
        #import matplotlib.pyplot as plt
        ## show images next to each other
        #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #ax[0].imshow(ref_sharp, cmap='gray')
        #ax[0].set_title('Reference')
        #ax[1].imshow(target_sharp, cmap='gray')
        #ax[1].set_title('Target')
        #plt.show()
    
        # Apply CLAHE for better contrast
        # CLAHE => https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#:~:text=Contrast%20Limited%20AHE%20(CLAHE)%20is,slope%20of%20the%20transformation%20function.
        clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        ref_clahe    = clahe.apply(ref_sharp)
        target_clahe = clahe.apply(target_sharp)

        ##DEBUGING
        #import matplotlib.pyplot as plt
        ## show images next to each other
        #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #ax[0].imshow(ref_clahe, cmap='gray')
        #ax[0].set_title('Reference')
        #ax[1].imshow(target_clahe, cmap='gray')
        #ax[1].set_title('Target')
        #plt.show()

        # Try multiple feature detection methods and use the best result
        methods            = self.featureMatchingMethods 
        best_homography    = None
        best_confidence    = 0
        best_inliers       = 0
        best_shift_x       = 0
        best_shift_y       = 0
        best_rotation      = 0
        best_skew          = (0, 0)
        best_method        = None
        best_stats         = None
        best_matches_count = 0

        for method in methods:
            try:
                if method == 'orb':
                    # ORB detector
                    detector = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
                elif method == 'sift':
                    # SIFT detector - better for transformations
                    detector = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04)
                elif method == 'akaze':
                    # AKAZE detector
                    detector = cv2.AKAZE_create()
                elif method == 'brisk':
                    # BRISK detector
                    detector = cv2.BRISK_create() 
                elif method == 'kaze':
                    # KAZE detector
                    detector = cv2.KAZE_create()
                elif method == 'brief':
                    # BRIEF detector
                    detector = cv2.xfeatures2d.BriefDescriptorExtractor_create()
                elif method == 'fast':
                    # FAST detector
                    detector = cv2.FastFeatureDetector_create()
                else:
                    print(f"Unknown feature detection method: {method}, \n    skipping...")
                    continue
                
                # Find keypoints and descriptors
                kp1, des1 = detector.detectAndCompute(ref_clahe,    None)
                kp2, des2 = detector.detectAndCompute(target_clahe, None)
                
                # Check if enough features were detected (at least 10 features must be present in both images)
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

                # Sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)

                # Take only good matches
                num_good_matches = int(len(matches) * good_ratio)
                good_matches     = matches[:num_good_matches]
                
                # Check if enough good matches were found (at least 10 good matches are required => Should be the same as the number of features in each image)
                if len(good_matches) < 10:
                    if verbose:
                        print(f"Not enough good matches found with {method}")
                    continue
                
                # Extract matched keypoints for homography calculation
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                
                # Find homography matrix
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is None:
                    if verbose:
                        print(f"Failed to find homography with {method}")
                    continue
                    
                # Count inliers
                inliers_count = np.sum(mask)
                
                # Check if enough inliers were found (at least 8 inliers are required)
                if inliers_count < 8:
                    if verbose:
                        print(f"Not enough inliers with {method}: {inliers_count}")
                    continue
                
                # Calculate confidence based on inlier ratio and number of matches
                inlier_ratio = inliers_count / len(good_matches)
                confidence   = inlier_ratio * min(1.0, len(good_matches) / 100)
                
                # Extract transformation components from homography matrix
                # Homography matrix has the form:
                # [ a b c ]
                # [ d e f ]
                # [ g h 1 ]
                
                # For pure similarity transformation (rotation + scale + translation),
                # the matrix would have a=e and b=-d, but we allow for skew as well
                
                # Image center point for rotation calculation
                height, width      = ref_clahe.shape
                center_x, center_y = width / 2, height / 2
                
                # Apply homography to center point to get translation
                center_point        = np.array([center_x, center_y, 1])
                transformed_center  = H.dot(center_point)
                transformed_center /= transformed_center[2]  # Normalize

                # Translation in pixels
                shift_x = transformed_center[0] - center_x
                shift_y = transformed_center[1] - center_y

                # Calculate rotation angle from the homography
                # For a rotation matrix [cos(θ) -sin(θ); sin(θ) cos(θ)],
                # the angle is θ = atan2(sin(θ), cos(θ)) = atan2(matrix[1,0], matrix[0,0])
                # For homography, we need to normalize by removing the perspective effect
                
                # Extract the 2x2 affine part of the homography matrix H
                affine_part = H[:2, :2]
                
                # Singular Value Decomposition (SVD) to separate rotation from scale and skew
                # U = left singular vectors, S = singular values, Vt = right singular vectors
                U, S, Vt = np.linalg.svd(affine_part)

                # The singular values S represent the scaling factors
                scale_x = S[0]
                scale_y = S[1]
                scale_x = np.clip(scale_x, 0.5, 2.0)
                scale_y = np.clip(scale_y, 0.5, 2.0)
                # If values are very close to 1, just return 1
                if abs(scale_x - 1.0) < 0.02:
                    scale_x = 1.0
                if abs(scale_y - 1.0) < 0.02:
                    scale_y = 1.0

                # Compute rotation matrix (proper rotation, no reflection)
                R = U @ Vt
                
                # Ensure it's a proper rotation matrix (det=1)
                if np.linalg.det(R) < 0:
                    R[:, -1] *= -1
                
                # Extract rotation angle in degrees
                rotation_angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                
                # Compute remaining transformation components (scale and skew)
                # Scale factors
                sx = np.linalg.norm(affine_part[:, 0])
                sy = np.linalg.norm(affine_part[:, 1])
                
                # Limit scale factors to reasonable values
                sx = np.clip(sx, 0.5, 2.0)
                sy = np.clip(sy, 0.5, 2.0)

                # Skew angles
                skew_x = np.degrees(np.arctan2(affine_part[0, 1], affine_part[1, 1]))
                skew_y = np.degrees(np.arctan2(affine_part[1, 0], affine_part[0, 0])) - 90
                
                # Calculate quality metrics for the transformation
                skew_magnitude = np.sqrt(skew_x**2 + skew_y**2)

                # Penalize excessive skew or scale differences
                if abs(sx - sy) > 0.2 or skew_magnitude > 5.0:
                    confidence *= 0.8

                # Apply tuning weight on confidence
                confidence = confidence * float(self.config["ORTHORECTIFICATION"]["feature_weight"])

                # Update best result if this method performed better
                if confidence > best_confidence and inliers_count >= 8:
                    best_homography    = H
                    best_confidence    = confidence
                    best_inliers       = inliers_count
                    best_shift_x       = shift_x
                    best_shift_y       = shift_y
                    best_rotation      = rotation_angle
                    best_skew          = (skew_x, skew_y)
                    best_scale_factors = (sx, sy)
                    best_method        = method
                    best_matches_count = len(good_matches)

                    # Calculate statistics for compatibility with other methods
                    match_points     = np.hstack((src_pts, dst_pts))
                    shift_magnitudes = np.sqrt(np.sum((dst_pts - src_pts)**2, axis=1))
                    mean_magnitude   = np.mean(shift_magnitudes)
                    std_magnitude    = np.std(shift_magnitudes)
                    best_stats       = (confidence, mean_magnitude, std_magnitude)

                    if verbose:
                        print(f"Method {method}: shift=({shift_x:.2f}, {shift_y:.2f}), "
                              f"rotation={rotation_angle:.2f}°, skew=({skew_x:.2f}, {skew_y:.2f}), "
                              f"confidence={confidence:.2f}, inliers={inliers_count}/{len(good_matches)}")

                # Visualize matches (for debugging)
                if visualize:
                    # Draw matches
                    match_img = cv2.drawMatches(ref_clahe, kp1, target_clahe, kp2, good_matches, None)

                    # Draw homography transformation
                    h, w = ref_clahe.shape
                    pts  = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst  = cv2.perspectiveTransform(pts, H)

                    # Convert to int for drawing
                    dst_int       = np.int32(dst)
                    transform_img = cv2.polylines(match_img.copy(), [dst_int], True, (0, 255, 0), 3)

                    # Save images
                    cv2.imwrite(f"matches_{method}.jpg", match_img)
                    cv2.imwrite(f"transform_{method}.jpg", transform_img)

            except Exception as e:
                if verbose:
                    print(f"Error with {method} matching: {str(e)}")
                continue

        # If no method found good matches
        if best_confidence == 0:
            if verbose:
                print("No reliable transformation found with any feature matching method")
            return 0, 0, 0, (0, 0), (1, 1) ,None

        # Convert pixel shifts to meters using geotransform
        x_shift_m = best_shift_x * abs(target_img.GetGeoTransform()[1])
        y_shift_m = best_shift_y * abs(target_img.GetGeoTransform()[5])

        if verbose:
            print(f"Final transformation: shift=({x_shift_m:.4f}m, {y_shift_m:.4f}m), "
                  f"rotation={best_rotation:.2f}°, skew={best_skew}, "
                  f"confidence={best_confidence:.4f}, method={best_method}")

        return x_shift_m, y_shift_m, best_rotation, best_skew, best_scale_factors,  best_stats

    def _shift_opticalFlow_multiscale(self, ref_img, target_img, band=0, verbose=False, tuning_val_percentage=0.25)     -> tuple:
        """
        Calculate local shift between two overlapping images using a multi-scale approach.
        This handles larger shifts more effectively by working at different resolutions.

        INPUTS:
            ref_img (gdal.Dataset):  Reference image (image overlap + frame)
            target_img (gdal.Dataset):  Target image (image overlap + frame)
            band (int):  Band index to use (default: 0)
            verbose (bool):  Print debug messages in this function (default: False)
            tuning_val_percentage (float):  Minimum valid data percentage for optical flow (default: 0.25)

        OUTPUT:
            tuple: (x_shift, y_shift, confidence_stats)
        """
        pyramid_levels     = 4  # Number of pyramid levels for multi-scale approach
        min_no_inliners    = 10 # Minimum number of inliers required for a valid transformation
        inlier_threshold_factor = 2.5 # TODO: Description

        if ref_img is None or target_img is None:
            if verbose:
                print("Error: One of the input images is None")
            return 0, 0, 0, (0, 0), None

        ref_ds, target_ds, ref_nodata, target_nodata = self._prepare_images_for_processing(ref_img, target_img, band)
        
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

        ## DEBUGING
        #import matplotlib.pyplot as plt
        #plt.imshow(valid_mask)

        # Skip calculation if there's not enough valid data
        valid_percentage = np.sum(valid_mask) / valid_mask.size
        if valid_percentage < tuning_val_percentage:
            if verbose:
                print(f"Warning: Not enough valid data in overlap ({valid_percentage:.1%}). Skipping optical flow.")
            return 0, 0, 0, (0, 0), None
    
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
        cumulative_x_shift  = 0.0
        cumulative_y_shift  = 0.0
        cumulative_rotation = 0.0
        cumulative_skew_x   = 0.0
        cumulative_skew_y   = 0.0
        best_confidence     = 0.0
        best_stats          = None

        # Process from coarsest to finest
        for i in range(pyramid_levels - 1, -1, -1):
            scale_factor = 2**i
            
            ref_current        = ref_pyramid[i]
            target_current     = target_pyramid[i]
            valid_mask_current = valid_mask_pyramid[i]
            
            # Apply current cumulative shift to the target image (scaled appropriately)
            if i < pyramid_levels - 1:  # Skip first iteration on coarsest level
                h, w           = target_current.shape

                # Scale down the current transformations for that level
                x_shift_scaled  = cumulative_x_shift / scale_factor
                y_shift_scaled  = cumulative_y_shift / scale_factor
                rotation_scaled = cumulative_rotation # Rotation is not scaled
                skew_x_scaled   = cumulative_skew_x
                skew_y_scaled   = cumulative_skew_y

                # Create affine transformation matrix incorporating all transformations
                center = (w / 2, h / 2)

                # First create rotation matrix
                rotation_mat = cv2.getRotationMatrix2D(center, rotation_scaled, 1.0)

                # Add skew to the matrix
                skew_mat = np.float32([[1, np.tan(np.radians(skew_x_scaled)), 0],
                                      [np.tan(np.radians(skew_y_scaled)), 1, 0]])
            
                # Combine rotation and skew
                combined_mat         = np.matmul(rotation_mat[:2, :2], skew_mat[:2, :2])
                transform_mat        = np.zeros((2, 3), dtype=np.float32)
                transform_mat[:, :2] = combined_mat
                transform_mat[0, 2]  = x_shift_scaled + rotation_mat[0, 2]
                transform_mat[1, 2]  = y_shift_scaled + rotation_mat[1, 2]

                # Apply the full transformation
                target_current = cv2.warpAffine(target_current, transform_mat, (w, h), 
                                                borderMode=cv2.BORDER_REPLICATE)
            
            #######################################
            # Calculate optical flow at this level
            #######################################
            try:
                # Parameters adjusted for the scale
                win_size   = max(15, int(21 / (i + 1)))
                iterations = max(5, int(10 / (i + 1)))
                
                flow = cv2.calcOpticalFlowFarneback(
                    ref_current, 
                    target_current, 
                    None,
                    pyr_scale  = 0.5,                          # Scale factor for the pyramid (0.5 = half size)
                    levels     = min(5, pyramid_levels - i),  # Fewer levels at smaller scales
                    winsize    = win_size,
                    iterations = iterations,
                    poly_n     = 5,                           # Polynomial expansion for interpolation (higher = more stable)
                    poly_sigma = 1.1,                         # Standard deviation of the Gaussian used to smooth derivatives
                    flags      = 0                            # No additional flags  
                )
                
                # Extract flow components
                x_flow = flow[:, :, 0]
                y_flow = flow[:, :, 1]
                
                # Only consider valid regions for flow statistics
                flow_valid = valid_mask_current.copy()
                x_valid    = x_flow[flow_valid]
                y_valid    = y_flow[flow_valid]
                
                if len(x_valid) == 0 or len(y_valid) == 0:
                    continue

                # Estimate translation from the median flow (robust to outliers)
                level_x_shift = np.median(x_valid)
                level_y_shift = np.median(y_valid)

                # Extract rotation and skew from the flow field
                # We need coordinates of points in the image
                h, w           = ref_current.shape
                y_grid, x_grid = np.mgrid[0:h, 0:w]

                # Only consider valid points
                x_points = x_grid[flow_valid]
                y_points = y_grid[flow_valid]

                # Center coordinates around the image center
                center_x, center_y = w / 2, h / 2
                x_centered         = x_points - center_x
                y_centered         = y_points - center_y

                # End points after flow
                x_end = x_points + x_flow[flow_valid]
                y_end = y_points + y_flow[flow_valid]

                # Center the end points too
                x_end_centered = x_end - center_x
                y_end_centered = y_end - center_y

                # Filter out the points that moved too much (outliers)
                flow_magnitudes  = np.sqrt((x_end - x_points)**2 + (y_end - y_points)**2)
                median_magnitude = np.median(flow_magnitudes)
                inlier_threshold = median_magnitude * inlier_threshold_factor
                inliers          = flow_magnitudes < inlier_threshold

                if np.sum(inliers) < min_no_inliners:
                    # Skip roation and skew calculation if not enough inliers
                    level_rotation = 0.0
                    level_skew_x   = 0.0
                    level_skew_y   = 0.0
                else:
                    # Use only inlier points
                    x_centered     = x_centered[inliers]
                    y_centered     = y_centered[inliers]
                    x_end_centered = x_end_centered[inliers]
                    y_end_centered = y_end_centered[inliers]

                    # Determine affine transformation using least squares
                    # We're solving for the 2x2 affine matrix A in [x_end; y_end] = A * [x; y]
                    # Prepare matrices for least squares solution
                    P         = np.column_stack([x_centered, y_centered, np.ones_like(x_centered)])
                    x_end_vec = x_end_centered
                    y_end_vec = y_end_centered

                    # Solve the system for x and y coordinates separately
                    try:
                        # Use regularized least squares for stability
                        # Add a small identity matrix to avoid instability
                        PtP = P.T @ P + 1e-3 * np.eye(P.shape[1])
                        PtX = P.T @ x_end_vec
                        PtY = P.T @ y_end_vec

                        x_coeffs = np.linalg.solve(PtP, PtX)
                        y_coeffs = np.linalg.solve(PtP, PtY)

                        # Extract the 2x2 affine part
                        A = np.array([[x_coeffs[0], x_coeffs[1]],
                                      [y_coeffs[0], y_coeffs[1]]])

                        # Decompose into rotation, scale, and skew using SVD
                        U, S, Vt = np.linalg.svd(A)

                        # Extract scale factors from the singular values
                        # Scale can be extracted from the SVD of the affine matrix A
                        # S contains the singular values which represent scaling factors
                        sx = S[0]
                        sy = S[1]

                        # Ensure scale factors are reasonable
                        sx = np.clip(sx, 0.5, 2.0)
                        sy = np.clip(sy, 0.5, 2.0)

                        # Store scale factors
                        cumulative_scale_x = 1.0
                        cumulative_scale_y = 1.0
                        
                        # Add to variables tracking cumulative transformation
                        # Existing code for cumulative transformations...
                        
                        # In the calculations after optical flow:
                        # Add scale factor calculations
                        cumulative_scale_x *= sx
                        cumulative_scale_y *= sy
                        
                        # In the final return statement, include scale factors
                        img_scale_factors = (cumulative_scale_x, cumulative_scale_y)

                        # Compute rotation matrix (proper rotation, no reflection)
                        R = U @ Vt

                        # Ensure it's a proper rotation matrix (det=1)
                        if np.linalg.det(R) < 0:
                            U[:, -1] *= -1
                            R         = U @ Vt

                        # Extract rotation angle in degrees
                        level_rotation = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

                        # Compute skew angles
                        # For small angles, the skew can be approximated from the affine matrix
                        # Skew angles in degrees
                        level_skew_x = np.degrees(np.arctan2(A[0, 1], A[1, 1]))
                        level_skew_y = np.degrees(np.arctan2(A[1, 0], A[0, 0])) - 90

                        # Limit the transformation values to reasonable ranges
                        # This helps prevent extreme values from noise
                        max_rotation = 5.0 / scale_factor  # Allow larger rotations at coarser levels
                        max_skew     = 3.0 / scale_factor

                        level_rotation = np.clip(level_rotation, -max_rotation, max_rotation)
                        level_skew_x   = np.clip(level_skew_x, -max_skew, max_skew)
                        level_skew_y   = np.clip(level_skew_y, -max_skew, max_skew)

                    except np.linalg.LinAlgError:
                        # Fallback if the linear system is unstable
                        level_rotation = 0
                        level_skew_x   = 0
                        level_skew_y   = 0
                # Scale the transformation values to the original image size
                level_x_shift *= scale_factor
                level_y_shift *= scale_factor
                # Rotation and skew angles don't need scaling
                
                # Add to cumulative transformation
                cumulative_x_shift  += level_x_shift
                cumulative_y_shift  += level_y_shift
                cumulative_rotation += level_rotation
                cumulative_skew_x   += level_skew_x
                cumulative_skew_y   += level_skew_y
                
                # Calculate confidence based on flow consistency
                flow_magnitude = np.sqrt(x_valid**2 + y_valid**2)
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
                    confidence = min(confidence, 1.0)
                
                # Higher confidence for finer levels
                confidence *= (1.0 + (pyramid_levels - i - 1) * 0.1)
                confidence  = min(confidence, 1.0)
                
                # Reduce confidence if large rotation or skew was detected
                # (optical flow is less reliable for these)
                if abs(level_rotation) > 2.0 or abs(level_skew_x) > 1.0 or abs(level_skew_y) > 1.0:
                    confidence *= 0.8

                # Update confidence with tuning weight
                confidence = confidence * float(self.config["ORTHORECTIFICATION"]["opticalflow_weight"])

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_stats      = (confidence, mean_magnitude, std_magnitude)
                
                if verbose:
                    print(f"Level {i} (scale factor {scale_factor}):")
                    print(f"  Level shift: x={level_x_shift:.2f}, y={level_y_shift:.2f}")
                    print(f"  Level rotation: {level_rotation:.2f}°")
                    print(f"  Level skew: ({level_skew_x:.2f}, {level_skew_y:.2f})°")
                    print(f"  Cumulative: shift=({cumulative_x_shift:.2f}, {cumulative_y_shift:.2f}), "
                          f"rot={cumulative_rotation:.2f}°, skew=({cumulative_skew_x:.2f}, {cumulative_skew_y:.2f})°")
                    print(f"  Confidence: {confidence:.2f}")
                
            except Exception as e:
                if verbose:
                    print(f"Error at pyramid level {i}: {str(e)}")
                continue
    
        # Apply the geotransform to calculate actual shifts in meters
        x_shift_m = cumulative_x_shift * abs(target_img.GetGeoTransform()[1])
        y_shift_m = cumulative_y_shift * abs(target_img.GetGeoTransform()[5])
        
        # Rotation and skew are in degrees already
        rotation_angle = cumulative_rotation
        skew_params = (cumulative_skew_x, cumulative_skew_y)
        
        if verbose:
            print(f"Final transformation: shift=({x_shift_m:.4f}m, {y_shift_m:.4f}m), "
                  f"rotation={rotation_angle:.2f}°, skew={skew_params}°")
            print(f"Confidence: {best_confidence:.4f}")
        
        # If no valid transformation was found
        if best_stats is None:
            return 0, 0, 0, (0, 0), None
        
        return x_shift_m, y_shift_m, rotation_angle, skew_params, img_scale_factors, best_stats
    
    def _shift_phaseCorrelation(self, ref_img, target_img, band=0, verbose=False)                                       -> tuple:
        """
        Calculate shift between two images using phase correlation.
        This frequency-domain approach can detect large shifts and is rotation-invariant.

        INPUTS:
        - ref_img (gdal.Dataset):     Reference image (image overlap + frame)
        - target_img (gdal.Dataset):  Target image (image overlap + frame)
        - band (int):                 Band index to use (default: 0)
        - verbose (bool):             Print debug messages in this function (default: False)

        OUTPUT:
        - tuple: (x_shift, y_shift, confidence_stats)

        NOTE: Confidence is caluclated based on the "strength" of the phase cross-correlation peak.
        => Higher peak values indicate a more reliable shift estimation -> higher confidence.
        => Confidence is normalized to [0, 1] based on the peak value relative to the maximum.
        => Tuning value: "tuning_val_confidence_percentage" (default: 0.3, 30% of the peak value assumed to be "max" (due to noise))
        """
        tuning_val_confidence_percentage = 0.3
        min_valid_data                   = 0.25 # Minimum valid data percentage for phase correlation

        if ref_img is None or target_img is None:
            if verbose:
                print("Error: One of the input images is None")
            return 0, 0, None

        ref_ds, target_ds, ref_nodata, target_nodata = self._prepare_images_for_processing(ref_img, target_img, band)

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

        ## DEBUGING
        #import matplotlib.pyplot as plt
        #plt.imshow(valid_mask)

        # Skip calculation if there's not enough valid data
        valid_percentage = np.sum(valid_mask) / valid_mask.size
        if valid_percentage < min_valid_data:
            if verbose:
                print(f"Warning: Not enough valid data in overlap ({valid_percentage:.1%}). Skipping phase correlation.")
            return 0, 0, None

        # Replace invalid pixels with mean value
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

        # Convert to float32 and normalize
        ref_float    = ref_ds.astype(np.float32)
        target_float = target_ds.astype(np.float32)

        # Apply windowing to reduce edge effects (Hann window)
        h, w            = ref_float.shape
        y, x            = np.ogrid[:h, :w]
        window          = np.hanning(h)[:, np.newaxis] * np.hanning(w)
        ref_windowed    = ref_float * window
        target_windowed = target_float * window

        # Perform phase correlation at multiple scales #NOTE: Tuning value
        scales        = [1.0, 0.5, 0.25, 0.125]  # Try at original size and two downsampled versions

        # Initialize best shift values
        best_shift_x  = 0
        best_shift_y  = 0
        best_response = 0
        best_scale    = 1.0

        for scale in scales:
            try:
                # Skip very small images
                if min(h * scale, w * scale) < 32:
                    continue

                # Resize if not at original scale
                if scale < 1.0:
                    h_scaled      = max(int(h * scale), 32)
                    w_scaled      = max(int(w * scale), 32)
                    ref_scaled    = cv2.resize(ref_windowed, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
                    target_scaled = cv2.resize(target_windowed, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
                else:
                    ref_scaled    = ref_windowed
                    target_scaled = target_windowed

                # Laplacian kernel (high-pass filter) for edge enhancement
                kernel    = np.array([[-1, -1, -1], 
                                      [-1,  8, -1], 
                                      [-1, -1, -1]]) / 8.0
                ref_hp    = cv2.filter2D(ref_scaled,    -1, kernel)
                target_hp = cv2.filter2D(target_scaled, -1, kernel)

                # Compute FFT of both images
                fft_ref    = np.fft.fft2(ref_hp)
                fft_target = np.fft.fft2(target_hp)

                # Compute cross-power spectrum
                cross_power     = fft_ref * np.conj(fft_target)
                cross_power_abs = np.abs(cross_power) + 1e-10  # Add small epsilon to avoid division by zero

                # Compute normalized cross-power spectrum (phase correlation)
                r = cross_power / cross_power_abs

                # Compute inverse FFT
                result     = np.fft.ifft2(r)
                result_abs = np.abs(result)

                # Find peak location
                idx          = np.unravel_index(np.argmax(result_abs), result_abs.shape)
                max_response = result_abs[idx]

                # Convert to shift values (considering FFT's shift property)
                h_scaled, w_scaled = result_abs.shape
                y_shift            = idx[0]
                x_shift            = idx[1]

                # Adjust shifts that are beyond the half-size (FFT wrap-around)
                if y_shift > h_scaled // 2:
                    y_shift -= h_scaled
                if x_shift > w_scaled // 2:
                    x_shift -= w_scaled

                # Scale shifts back to original image size
                x_shift_orig = x_shift / scale
                y_shift_orig = y_shift / scale

                # Check if this scale gave a better response
                if max_response > best_response:
                    best_response = max_response
                    best_shift_x  = x_shift_orig
                    best_shift_y  = y_shift_orig
                    best_scale    = scale

                    if verbose:
                        print(f"Scale {scale}: shift=({x_shift_orig:.2f}, {y_shift_orig:.2f}), response={max_response:.4f}")

            except Exception as e:
                if verbose:
                    print(f"Error in phase correlation at scale {scale}: {str(e)}")
                continue

        if best_response == 0:
            if verbose:
                print("Phase correlation failed to find a reliable shift")
            return 0, 0, None

        # Calculate confidence from response value (normalize between 0 and 1)
        # Phase correlation peak height is a good indicator of match quality
        confidence = min(best_response / tuning_val_confidence_percentage, 1.0)  # Normalize with typical max value

        x_shift_m = best_shift_x * abs(target_img.GetGeoTransform()[1])
        y_shift_m = best_shift_y * abs(target_img.GetGeoTransform()[5])

        # Generate statistics similar to other methods
        mean_magnitude = np.sqrt(best_shift_x**2 + best_shift_y**2)

        # Caluculate standard deviation of the shift magnitudes
        std_deviation = 0.0
        if mean_magnitude > 0:
            # Get a region around the correlation peak
            region_size = 20
            y_min = max(0, idx[0] - region_size)
            y_max = min(result_abs.shape[0], idx[0] + region_size + 1)
            x_min = max(0, idx[1] - region_size)
            x_max = min(result_abs.shape[1], idx[1] + region_size + 1)

            # Calculate standard deviation of values in the peak region
            # Higher std means less sharply defined peak (less certainty)
            peak_region = result_abs[y_min:y_max, x_min:x_max]

            # Calculate standard deviation of the region
            std_deviation = np.std(peak_region)

            # Normalize by the mean magnitude
            normalized_std = std_deviation / max_response

            # Convert to a confidence measure (lower std = higher confidence)
            std_deviation = normalized_std * mean_magnitude

        # Update confidence with tuning weight
        confidence = confidence * float(self.config["ORTHORECTIFICATION"]["phasecorrelation_weight"])
        stats = (confidence, mean_magnitude, std_deviation)

        if verbose:
            print(f"Final phase correlation shift in meters: x={x_shift_m:.4f}m, y={y_shift_m:.4f}m")
            print(f"Confidence: {confidence:.4f}, Peak response: {best_response:.4f}")
        
        return x_shift_m, y_shift_m, stats

# TODO implement NCC /alternative to Phase Correlation
# TODO implement RANSAC for feature matching ??
# TODO: Allow the choice to chose one (or a selection) of  methods used for orthorectification => config file with flags in "...hybrid"