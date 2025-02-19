import os
import numpy       as np
from tqdm             import tqdm
from osgeo            import gdal, osr
from pathlib          import Path
from scipy.signal     import correlate2d
from rasterio.windows import from_bounds
import tools.visualizations as vis
import rasterio             as rasterio
from rasterio import warp

#import tools.constants as c

class FeatureMatching:
    def __init__(self, config):
        self.config    = config
        self.image_dir = config["MISSION"]["outputfolder"]
        self.images    = None # Image dictionary containing all the image information
        self.offsets   = None

        self.load_geotiffs()

    def load_geotiffs(self):
        """
        A function for loading all the geotiff images in the image directory.
        """
        file_path = Path(self.image_dir)
        print("Loading images from", file_path)


        tif_files = list(file_path.glob("*.tif"))
        self.images = {}

        for file in tqdm(tif_files, desc="Loading Geotiffs"):
            try:
                # Create image information dirtionary for each image
                # Load the image using rasterio and store it in the dictionary (read-only)
#                src = rasterio.open(file, "r")
                src = gdal.Open(str(file))
                self.images[str(file)] = {
                    "filepath": str(file),
                    "gdalImg":  src
                }
            except Exception as e:
                print(f"Error loading {file}")
                print(e)

        print("Loaded", len(self.images), "images.")

    def process_image_grid(self):
        """Process images from NW to SE, calculating required offsets."""
        offsets        = {}
        processed_set  = set()
    
        # Convert dictionary values to a list for sorting
        image_list = list(self.images.values())

#        vis.plot_georeferenced_images(image_list, first_idx=0, last_idx=2, title='Georeferenced Mugin Images', cmap='terrain', figsize=(10, 8), show_overlap=True)

        # Look through all images 
        for i, img in enumerate(image_list):
            # Get the path of the image
            path = img["filepath"]
            # If it is the first image, add it to the processed set and continue
            if i == 0:
                processed_set.add(i)
                continue

            # Get image data of previous and current image
            base_img      = self.images[image_list[i-1]["filepath"]]['gdalImg']
            img           = self.images[image_list[i]["filepath"]]['gdalImg']

            # Get the image overlap
            overlap_n_1, overlap_n = self._get_overlap_dataset(image_list[i-1]["filepath"],  # Base image (n-1)
                                                               image_list[i]["filepath"])    # Current image (n)

#            vis.visualize_overlap(overlap_n_1, overlap_n)
            if (overlap_n_1 is None) or (overlap_n is None):
                print(f"WARNING: No overlap between {image_list[i-1]['filepath']} and {image_list[i]['filepath']}")
                continue

            a = 1
            # Get the image data of the image i

        for i, img in enumerate(image_list):
            path = img["filepath"]
            if i == 0:
                processed.add(path)
                continue

            img_data      = image_list[i]["gdalImg"]["bands"][0]   # Using the first band
            img_transform = image_list[i]["gdalImg"]["transform"]
            img_bounds    = image_list[i]["gdalImg"]["bounds"]

            # Find neighboring processed images
            for processed_path in processed:
                ref_img       = image_list[i]
                ref_data      = ref_img["bands"][0]  # Using first band
                ref_transform = ref_img["transform"]
                ref_bounds    = ref_img["bounds"]

                # Check if images are neighbors and have overlap
                overlap_bounds = self._find_overlap_bounds(ref_bounds, img_bounds)

                if overlap_bounds[2] > overlap_bounds[0] and overlap_bounds[3] > overlap_bounds[1]:
                    try:
                        # Extract overlapping regions
                        ref_overlap = self.extract_overlap(ref_data, ref_transform, overlap_bounds)
                        img_overlap = self.extract_overlap(img_data, img_transform, overlap_bounds)

                        if ref_overlap.size == 0 or img_overlap.size == 0:
                            continue

                        # Calculate offset using the "find_offset" function
                        y_offset, x_offset = self.find_offset(ref_overlap, img_overlap)
                        offsets[path]      = {
                                 "reference":  processed_path,
                                 "x_shift":    x_offset,
                                 "y_shift":    y_offset,
                                 "pixel_size": img_transform[0]
                        }

                    except Exception as e:
                        print(f"Error extracting overlap for images {processed_path} and {path}")
                        print(e)
        processed.add(path)
        self.offsets

    def _get_overlap_dataset(self, img_name1, img_name2):
        """
        Extracts the exact overlapping region between two images as shown in red in plot_georeferenced_images.
        
        INPUT: The "self.images" dictionary keys for the two images.
        OUTPUT: Two overlapping datasets with identical bounds and dimensions.
        """
        # Get the rasterio dataset objects for both images
        ds_1 = self.images[img_name1]["gdalImg"]
        ds_2 = self.images[img_name2]["gdalImg"]
        
        # Get geotransform and corners for both images (similar to plot_georeferenced_images)
        # First image corners
        corners_1 = [
            (0, 0),
            (ds_1.RasterXSize, 0),
            (ds_1.RasterXSize, ds_1.RasterYSize),
            (0, ds_1.RasterYSize)
        ]
        
        # Second image corners
        corners_2 = [
            (0, 0),
            (ds_2.RasterXSize, 0),
            (ds_2.RasterXSize, ds_2.RasterYSize),
            (0, ds_2.RasterYSize)
        ]
        
        # Convert to world coordinates using the transforms
        gt1 = ds_1.GetGeoTransform()
        gt2 = ds_2.GetGeoTransform()
        world_corners_1 = [(gt1[0] + corner[0] * gt1[1] + corner[1] * gt1[2],
                            gt1[3] + corner[0] * gt1[4] + corner[1] * gt1[5]) for corner in corners_1]
        world_corners_2 = [(gt2[0] + corner[0] * gt2[1] + corner[1] * gt2[2],
                            gt2[3] + corner[0] * gt2[4] + corner[1] * gt2[5]) for corner in corners_2]

#        world_corners_1 = [ds_1.xy(*corner) for corner in corners_1]
#        world_corners_2 = [ds_2.xy(*corner) for corner in corners_2]
        
        # Create paths for both images (like in plot_georeferenced_images)
        from matplotlib.path import Path
        path_1 = Path(world_corners_1)
        path_2 = Path(world_corners_2)
        
        # Calculate bounds of the intersection
        corners_x1, corners_y1 = zip(*world_corners_1)
        corners_x2, corners_y2 = zip(*world_corners_2)
        
        # Get the overlapping bounds
        overlap_bounds = (
            max(min(corners_x1), min(corners_x2)),  # west
            max(min(corners_y1), min(corners_y2)),  # south
            min(max(corners_x1), max(corners_x2)),  # east
            min(max(corners_y1), max(corners_y2))   # north
        )
        
        # Check if there is any overlap
        if (overlap_bounds[2] <= overlap_bounds[0] or 
            overlap_bounds[3] <= overlap_bounds[1]):
            print(f"No overlap between the images {img_name1} and {img_name2}")
            return None, None
        
        # Calculate dimensions for the output based on the first image's resolution
        pixel_size_x = abs(ds_1.transform.a)
        pixel_size_y = abs(ds_1.transform.e)
        
        width = int(round((overlap_bounds[2] - overlap_bounds[0]) / pixel_size_x))
        height = int(round((overlap_bounds[3] - overlap_bounds[1]) / pixel_size_y))
        
        # Create output transform
        output_transform = rasterio.transform.from_bounds(
            overlap_bounds[0], overlap_bounds[1],
            overlap_bounds[2], overlap_bounds[3],
            width, height
        )
        
        # Setup output files
        output_dir = "C:/DocumentsLocal/07_Code/SeaBee/SeaBee_georef_seagulls/DATA"
        os.makedirs(output_dir, exist_ok=True)
        memfile_1_path = f"{output_dir}/overlap1.tif"
        memfile_2_path = f"{output_dir}/overlap2.tif"
        
        # Remove existing files
        for path in [memfile_1_path, memfile_2_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # Create output profile
        output_profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': ds_1.count,
            'dtype': ds_1.dtypes[0],
            'crs': ds_1.crs,
            'transform': output_transform,
            'nodata': ds_1.nodata
        }
        
        # Create output datasets
        with rasterio.open(memfile_1_path, 'w', **output_profile) as dst1, \
             rasterio.open(memfile_2_path, 'w', **output_profile) as dst2:
            
            # Process each band
            for band_idx in range(1, ds_1.count + 1):
                # Initialize output arrays
                nodata_value = ds_1.nodata if ds_1.nodata is not None else 0
                dst_array1 = np.full((height, width), nodata_value, dtype=output_profile['dtype'])
                dst_array2 = np.full((height, width), nodata_value, dtype=output_profile['dtype'])
                
                # Use exact reprojection with the overlap bounds
                rasterio.warp.reproject(
                    source=rasterio.band(ds_1, band_idx),
                    destination=dst_array1,
                    src_transform=ds_1.transform,
                    src_crs=ds_1.crs,
                    dst_transform=output_transform,
                    dst_crs=ds_1.crs,
                    resampling=rasterio.warp.Resampling.nearest
                )
                
                rasterio.warp.reproject(
                    source=rasterio.band(ds_2, band_idx),
                    destination=dst_array2,
                    src_transform=ds_2.transform,
                    src_crs=ds_2.crs,
                    dst_transform=output_transform,
                    dst_crs=ds_2.crs,
                    resampling=rasterio.warp.Resampling.nearest
                )
                
                # Write to output files
                dst1.write(dst_array1, band_idx)
                dst2.write(dst_array2, band_idx)
        
        return rasterio.open(memfile_1_path), rasterio.open(memfile_2_path)

    def _get_overlap_dataset2(self, img_name1, img_name2):
        """
        A function for extracting the exact overlapping region between two images,
        with original RGB colors preserved.
        INPUT: The "self.images" dictionary keys for the two images.
        OUTPUT: Two overlapping datasets with identical bounds and dimensions.
        """
        # Get the rasterio dataset objects for both images
        ds_1 = self.images[img_name1]["gdalImg"]
        ds_2 = self.images[img_name2]["gdalImg"]
    
        # Get all the corners of the images (pixel coordinates)
        corners_1 = [
            (0, 0),
            (ds_1.width, 0),
            (ds_1.width, ds_1.height),
            (0, ds_1.height)
        ]
        corners_2 = [
            (0, 0),
            (ds_2.width, 0),
            (ds_2.width, ds_2.height),
            (0, ds_2.height)
        ]
    
        # Get the world corners
        world_corners_1 = [ds_1.xy(*corner) for corner in corners_1]
        world_corners_2 = [ds_2.xy(*corner) for corner in corners_2]
    
        # Make sure the projections are the same
        if not ds_1.crs == ds_2.crs:
            raise ValueError(f"The CRS of the two images are not the same: {ds_1.crs} vs {ds_2.crs}")
        
        bounds_1 = self._get_true_bounds(world_corners_1)
        bounds_2 = self._get_true_bounds(world_corners_2)
    
        # Check if there is any overlap
        if (bounds_1[2] <= bounds_2[0] or bounds_2[2] <= bounds_1[0] or
            bounds_1[3] <= bounds_2[1] or bounds_2[3] <= bounds_1[1]):
            print(f"No overlap between the images {img_name1} and {img_name2}")
            return None, None  # No overlap
    
        # Calculate precise overlapping area - rounded to nearest pixel to ensure exact boundary
        overlap_bounds = (
            max(bounds_1[0], bounds_2[0]),  # west
            max(bounds_1[1], bounds_2[1]),  # south
            min(bounds_1[2], bounds_2[2]),  # east
            min(bounds_1[3], bounds_2[3])   # north
        )
        
        # Output directory
        output_dir = "C:/DocumentsLocal/07_Code/SeaBee/SeaBee_georef_seagulls/DATA"
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the finest resolution from either image for output
        res_1 = ds_1.res
        res_2 = ds_2.res
        output_res = (min(abs(res_1[0]), abs(res_2[0])), min(abs(res_1[1]), abs(res_2[1])))
        
        # Calculate dimensions precisely, ensuring we capture the exact bounds
        width = max(1, int(round((overlap_bounds[2] - overlap_bounds[0]) / output_res[0])))
        height = max(1, int(round((overlap_bounds[3] - overlap_bounds[1]) / output_res[1])))
        
        # Recalculate bounds to ensure perfect alignment with pixel grid
        output_transform = rasterio.transform.from_bounds(
            *overlap_bounds, width=width, height=height
        )
        
        memfile_1_path = f"{output_dir}/overlap1.tif"
        memfile_2_path = f"{output_dir}/overlap2.tif"
        
        # Remove existing files if they exist
        for path in [memfile_1_path, memfile_2_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    print(f"Could not remove existing file: {path}")
        
        # Determine number of bands to preserve
        band_count = min(ds_1.count, ds_2.count)
        if band_count < 1:
            print(f"Error: No valid bands found in one of the images")
            return None, None
        
        # Create output profile with original bands and data type
        output_profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': band_count,
            'dtype': ds_1.dtypes[0],  # Use data type from first image
            'crs': ds_1.crs,
            'transform': output_transform,
            'nodata': ds_1.nodata
        }
        
        # Create both output datasets
        with rasterio.open(memfile_1_path, 'w', **output_profile) as dst1, \
             rasterio.open(memfile_2_path, 'w', **output_profile) as dst2:
            
            # Process each band
            for band_idx in range(1, band_count + 1):
                # Create destination arrays
                dst_array1 = np.zeros((height, width), dtype=output_profile['dtype'])
                dst_array2 = np.zeros((height, width), dtype=output_profile['dtype'])
                
                # Reproject data from both sources to the output with exact bounds
                rasterio.warp.reproject(
                    source=rasterio.band(ds_1, band_idx),
                    destination=dst_array1,
                    src_transform=ds_1.transform,
                    src_crs=ds_1.crs,
                    dst_transform=output_transform,
                    dst_crs=ds_1.crs,
                    resampling=rasterio.warp.Resampling.nearest
                )
                
                rasterio.warp.reproject(
                    source=rasterio.band(ds_2, band_idx),
                    destination=dst_array2,
                    src_transform=ds_2.transform,
                    src_crs=ds_2.crs,
                    dst_transform=output_transform,
                    dst_crs=ds_1.crs,
                    resampling=rasterio.warp.Resampling.nearest
                )
                
                # Write to output - preserving original data values
                dst1.write(dst_array1, band_idx)
                dst2.write(dst_array2, band_idx)
        
        # Return opened datasets
        return rasterio.open(memfile_1_path), rasterio.open(memfile_2_path)

    @staticmethod
    def _get_true_bounds(corners):
        """Helper function to get the true bounds from a set of corners"""
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    @staticmethod
    def _world_bounds_to_window(dataset, bounds):
        """Convert world bounds to pixel window for potentially rotated datasets"""
        # Calculate pixel coordinates for each corner of the bounds
        ul = dataset.index(bounds[0], bounds[3])  # Upper left
        ur = dataset.index(bounds[2], bounds[3])  # Upper right
        lr = dataset.index(bounds[2], bounds[1])  # Lower right
        ll = dataset.index(bounds[0], bounds[1])  # Lower left
        
        # Get the min/max row/col values
        rows = [ul[0], ur[0], lr[0], ll[0]]
        cols = [ul[1], ur[1], lr[1], ll[1]]
        
        # Create window from these bounds
        window = rasterio.windows.Window(
            col_off=max(0, min(cols)),
            row_off=max(0, min(rows)),
            width=max(cols) - min(cols),
            height=max(rows) - min(rows)
        )
        
        return window

    @staticmethod
    def extract_overlap(img_data, transform, overlap_bounds):
        """
        A function for extracting the overlapping area of the images.
        """
        rows, cols = rasterio.transform.rowcol(transform,
                                              [overlap_bounds[0], overlap_bounds[2]],
                                              [overlap_bounds[1], overlap_bounds[3]])

        return img_data[min(rows):max(rows), min(cols):max(cols)]

    @staticmethod
    def find_offset(img1, img2):
        """Find the offset between two images using correlation."""
        correlation = correlate2d(img1, img2, mode='full')
        y_max, x_max = np.unravel_index(correlation.argmax(), correlation.shape)
    
        y_offset = y_max - (img1.shape[0] - 1)
        x_offset = x_max - (img1.shape[1] - 1)
    
        return y_offset, x_offset