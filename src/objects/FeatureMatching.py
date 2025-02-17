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

#        no_files = len(list(file_path.glob("*.tif")))
        for file in tqdm(tif_files, desc="Loading Geotiffs"):
            try:
                # Create image information dirtionary for each image
                # Load the image using rasterio and store it in the dictionary (read-only)
                src = rasterio.open(file, "r")
                self.images[str(file)] = {
                    "filepath": str(file),
                    "rastImg":  src
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
            base_img      = self.images[image_list[i-1]["filepath"]]['rastImg']
            img           = self.images[image_list[i]["filepath"]]['rastImg']

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

            img_data      = image_list[i]["rastImg"]["bands"][0]   # Using the first band
            img_transform = image_list[i]["rastImg"]["transform"]
            img_bounds    = image_list[i]["rastImg"]["bounds"]

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
        A function for extracting the overlapping dataset between two images,
        returning only the overlap (absolute difference between overlapping regions).
        INPUT: The "self.images" dictionary keys for the two images.
        OUTPUT: Two overlapping datasets highlighting differences.
        """
        # Get the rasterio dataset objects for both images
        ds_1 = self.images[img_name1]["rastImg"]
        ds_2 = self.images[img_name2]["rastImg"]
    
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
    
        # Make sure the UTM projection is the same for both images
        if not ds_1.crs == ds_2.crs:
            raise ValueError(f"The CRS of the two images are not the same: {ds_1.crs} vs {ds_2.crs}")
        
        bounds_1 = self._get_true_bounds(world_corners_1)
        bounds_2 = self._get_true_bounds(world_corners_2)
    
        if (bounds_1[2] < bounds_2[0] or bounds_2[2] < bounds_1[0] or
            bounds_1[3] < bounds_2[1] or bounds_2[3] < bounds_1[1]):
            print(f"No overlap between the images {img_name1} and {img_name2}")
            return None, None  # No overlap
    
        # Calculate overlapping area
        overlap_bounds = (
            max(bounds_1[0], bounds_2[0]),  # west
            max(bounds_1[1], bounds_2[1]),  # south
            min(bounds_1[2], bounds_2[2]),  # east
            min(bounds_1[3], bounds_2[3])   # north
        )
        
        # Output directory
        output_dir = "C:/DocumentsLocal/07_Code/SeaBee/SeaBee_georef_seagulls/DATA"
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine the output resolution (use the finest resolution from either image)
        res_1 = ds_1.res
        res_2 = ds_2.res
        output_res = min(res_1[0], res_2[0]), min(res_1[1], res_2[1])
        
        # Calculate output dimensions
        width = int((overlap_bounds[2] - overlap_bounds[0]) / output_res[0])
        height = int((overlap_bounds[3] - overlap_bounds[1]) / output_res[1])
        
        # Ensure minimum dimensions
        width = max(width, 1)
        height = max(height, 1)
        
        memfile_1_path = f"{output_dir}/overlap1.tif"
        memfile_2_path = f"{output_dir}/overlap2.tif"
        
        # Remove existing files if they exist
        for path in [memfile_1_path, memfile_2_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    print(f"Could not remove existing file: {path}")
        
        # Create output transform
        output_transform = rasterio.transform.from_bounds(
            *overlap_bounds, width=width, height=height
        )
        
        # Create output profile with original data type
        output_profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': ds_1.count,  # Keep same band count as input
            'dtype': ds_1.dtypes[0],  # Use dtype from first band
            'crs': ds_1.crs,
            'transform': output_transform,
            'nodata': ds_1.nodata
        }

        # Create temporary arrays for warped images
        temp_data1 = np.zeros((ds_1.count, height, width), dtype=np.float32)
        temp_data2 = np.zeros((ds_2.count, height, width), dtype=np.float32)

        # Reproject each band from source images to temporary arrays
        for band in range(1, ds_1.count + 1):
            rasterio.warp.reproject(
                source=rasterio.band(ds_1, band),
                destination=temp_data1[band-1],
                src_transform=ds_1.transform,
                src_crs=ds_1.crs,
                dst_transform=output_transform,
                dst_crs=ds_1.crs,
                resampling=rasterio.warp.Resampling.nearest
            )
        
        for band in range(1, ds_2.count + 1):
            if band <= ds_2.count:
                rasterio.warp.reproject(
                    source=rasterio.band(ds_2, band),
                    destination=temp_data2[band-1],
                    src_transform=ds_2.transform,
                    src_crs=ds_2.crs,
                    dst_transform=output_transform,
                    dst_crs=ds_1.crs,
                    resampling=rasterio.warp.Resampling.nearest
                )
        
        # Create RGB outputs to highlight the differences
        with rasterio.open(memfile_1_path, 'w', **output_profile) as dst1, \
             rasterio.open(memfile_2_path, 'w', **output_profile) as dst2:
            
            # Create a mask for valid data in both images (where data is not 0 or nodata)
            mask1 = np.any(temp_data1 != 0, axis=0) & np.any(temp_data1 != ds_1.nodata, axis=0) if ds_1.nodata is not None else np.any(temp_data1 != 0, axis=0)
            mask2 = np.any(temp_data2 != 0, axis=0) & np.any(temp_data2 != ds_2.nodata, axis=0) if ds_2.nodata is not None else np.any(temp_data2 != 0, axis=0)
            
            # Find overlap area - where both masks are True
            overlap_mask = mask1 & mask2
            
            # Create normalized versions for comparison (using band 1 as reference)
            if np.any(overlap_mask):
                band1_data1 = temp_data1[0][overlap_mask]
                band1_data2 = temp_data2[0][overlap_mask]
                
                if band1_data1.max() > 0:
                    norm_factor1 = 255.0 / band1_data1.max()
                else:
                    norm_factor1 = 1.0
                    
                if band1_data2.max() > 0:
                    norm_factor2 = 255.0 / band1_data2.max()
                else:
                    norm_factor2 = 1.0
            else:
                norm_factor1 = 1.0
                norm_factor2 = 1.0
            
            # Create RGB outputs that highlight the overlap regions
            rgb1 = np.zeros((3, height, width), dtype=np.uint8)
            rgb2 = np.zeros((3, height, width), dtype=np.uint8)
            
            # For image 1: show normal grayscale with red in overlap area
            rgb1[0] = (temp_data1[0] * norm_factor1).astype(np.uint8)  # Red channel
            if ds_1.count > 1:
                rgb1[1] = (temp_data1[1] * norm_factor1).astype(np.uint8)  # Green channel
            else:
                rgb1[1] = (temp_data1[0] * norm_factor1).astype(np.uint8)  # Use band 1 for green
            if ds_1.count > 2:
                rgb1[2] = (temp_data1[2] * norm_factor1).astype(np.uint8)  # Blue channel
            else:
                rgb1[2] = (temp_data1[0] * norm_factor1).astype(np.uint8)  # Use band 1 for blue
            
            # For image 2: show normal grayscale with red in overlap area
            rgb2[0] = (temp_data2[0] * norm_factor2).astype(np.uint8)  # Red channel
            if ds_2.count > 1:
                rgb2[1] = (temp_data2[1] * norm_factor2).astype(np.uint8)  # Green channel
            else:
                rgb2[1] = (temp_data2[0] * norm_factor2).astype(np.uint8)  # Use band 1 for green
            if ds_2.count > 2:
                rgb2[2] = (temp_data2[2] * norm_factor2).astype(np.uint8)  # Blue channel
            else:
                rgb2[2] = (temp_data2[0] * norm_factor2).astype(np.uint8)  # Use band 1 for blue
                
            # Highlight overlap areas in red
            rgb1[0][overlap_mask] = 255  # Make red channel bright in overlap areas
            rgb2[0][overlap_mask] = 255  # Make red channel bright in overlap areas
            
            # Write to output files
            dst1.write(rgb1)
            dst2.write(rgb2)
        
        # Return opened datasets
        return rasterio.open(memfile_1_path), rasterio.open(memfile_2_path)

    def _get_overlap_dataset3(self, img_name1, img_name2):
        """
        A function for extracting the overlapping dataset between two images,
        returning only the red overlap (absolute difference between overlapping regions).
        INPUT: The "self.images" dictionary keys for the two images.
        OUTPUT: Two overlapping datasets highlighting differences.
        """
        # Get the rasterio dataset objects for both images
        ds_1 = self.images[img_name1]["rastImg"]
        ds_2 = self.images[img_name2]["rastImg"]

        # Get all the corners of the images
        corners_1 = [
            (0,0),
            (ds_1.width, 0),
            (ds_1.width, ds_1.height),
            (0, ds_1.height)
        ]
        corners_2 = [
            (0,0),
            (ds_2.width, 0),
            (ds_2.width, ds_2.height),
            (0, ds_2.height)
        ]

        # Get the world corners
        world_corners_1 = [ds_1.xy(*corner) for corner in corners_1]
        world_corners_2 = [ds_2.xy(*corner) for corner in corners_2]

        # make sure the UTM projection is the same for both images
        if not ds_1.crs == ds_2.crs:
            raise ValueError("The transformations of the two images are not the same.")
        
        bounds_1 = self._get_true_bounds(world_corners_1)
        bounds_2 = self._get_true_bounds(world_corners_2)

        if (bounds_1[2] < bounds_2[0] or bounds_2[2] < bounds_1[0] or
            bounds_1[3] < bounds_2[1] or bounds_2[3] < bounds_1[1]):
            print(f"No overlap between the images {img_name1} and {img_name2}")
            return None, None  # No overlap

        # Calculate overlapping area
        overlap_bounds = (
            max(bounds_1[0], bounds_2[0]),  # west
            max(bounds_1[1], bounds_2[1]),  # south
            min(bounds_1[2], bounds_2[2]),  # east
            min(bounds_1[3], bounds_2[3])   # north
        )

        # Calculate pixel dimensions for output
        width = int((overlap_bounds[2] - overlap_bounds[0]) / ds_1.transform.a)
        height = int((overlap_bounds[3] - overlap_bounds[1]) / abs(ds_1.transform.e))
    
        # Ensure minimum dimensions
        width = max(width, 1)
        height = max(height, 1)

        # Create output datasets with the same CRS
        output_transform = rasterio.transform.from_bounds(
            *overlap_bounds, width=width, height=height
        )

#        output_transform = rasterio.transform.from_bounds(
#            *overlap_bounds, 
#            width=int((overlap_bounds[2] - overlap_bounds[0]) / ds_1.transform.a), 
#            height=int((overlap_bounds[3] - overlap_bounds[1]) / abs(ds_1.transform.e))
#        )

        # Calculate pixel windows for each dataset
        window_1 = self._world_bounds_to_window(ds_1, overlap_bounds)
        window_2 = self._world_bounds_to_window(ds_2, overlap_bounds)
    
        # Read the data
        data_1 = ds_1.read(window=window_1)
        data_2 = ds_2.read(window=window_2)
    
        # Reproject both datasets to the same grid
        dst_height = max(window_1.height, window_2.height)
        dst_width = max(window_1.width, window_2.width)

        output_dir = "C:/DocumentsLocal/07_Code/SeaBee/SeaBee_georef_seagulls/DATA"
        os.makedirs(output_dir, exist_ok=True)

        # Create output datasets
        output_profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 3,  # RGB output
            'dtype': np.uint8,  # For visualization
            'crs': ds_1.crs,
            'transform': output_transform,
            'nodata': 0
        }
        
        memfile_1_path = f"{output_dir}/overlap1.tif"
        memfile_2_path = f"{output_dir}/overlap2.tif"
        
        # Close any existing files if they exist
        for path in [memfile_1_path, memfile_2_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    print(f"Could not remove existing file: {path}")
        
        # Create new output files
        with rasterio.open(memfile_1_path, 'w', **output_profile) as dst1, \
             rasterio.open(memfile_2_path, 'w', **output_profile) as dst2:
            
            # For each band in the image
            for band in range(1, min(ds_1.count, ds_2.count) + 1):
                # Reproject data from both sources to the output
                data1 = np.zeros((height, width), dtype=np.uint8)
                data2 = np.zeros((height, width), dtype=np.uint8)
            
                # Reproject band by band
                rasterio.warp.reproject(
                    source=rasterio.band(ds_1, band),
                    destination=data1,
                    src_transform=ds_1.transform,
                    src_crs=ds_1.crs,
                    dst_transform=output_transform,
                    dst_crs=ds_1.crs,
                    resampling=rasterio.warp.Resampling.nearest
                )
                
                rasterio.warp.reproject(
                    source=rasterio.band(ds_2, band),
                    destination=data2,
                    src_transform=ds_2.transform,
                    src_crs=ds_2.crs,
                    dst_transform=output_transform,
                    dst_crs=ds_1.crs,
                    resampling=rasterio.warp.Resampling.nearest
                )
            
                # Scale data for visualization (you may need to adjust this)
                if data1.max() > 0:
                    data1 = (data1 / data1.max() * 255).astype(np.uint8)
                if data2.max() > 0:
                    data2 = (data2 / data2.max() * 255).astype(np.uint8)
                
                # Write to all three bands for RGB visualization
                for out_band in range(1, 4):
                    dst1.write(data1, out_band)
                    dst2.write(data2, out_band)

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