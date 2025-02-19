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
            #overlap_n_1, overlap_n = self._get_overlap_dataset(image_list[i-1]["filepath"],  # Base image (n-1)
            #                                                   image_list[i]["filepath"],    # Current image (n)
            #                                                   save_overlap=True,
            #                                                   save_path="C:\\DocumentsLocal\\07_Code\\SeaBee\\SeaBee_georef_seagulls\\DATA\\")    
            overlap_n_1, overlap_n = self._get_overlap_dataset(image_list[i-1]["filepath"],  # Base image (n-1)
                                                               image_list[i]["filepath"],    # Current image (n)
                                                               save_overlap=False)    

            if (overlap_n_1 is None) or (overlap_n is None):
                print(f"WARNING: No overlap between {image_list[i-1]['filepath']} and {image_list[i]['filepath']}")
                continue
    
            # Get the image data of the image i
            calc_shift = self._calculate_shift(overlap_n_1, overlap_n)

            # Apply the shift to the image


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

    def _get_overlap_dataset(self, img_name1, img_name2, save_overlap=False, save_path=None):
        """
        Extracts the exact overlapping region between two images as shown in red in plot_georeferenced_images.
        
        INPUT: The "self.images" dictionary keys for the two images.
        OUTPUT: Two overlapping datasets with identical bounds and dimensions.
        """
        # Get the gdal dataset for both images
        ds_1 = self.images[img_name1]["gdalImg"]
        ds_2 = self.images[img_name2]["gdalImg"]
        
        # Get geotransforms
        gt1 = ds_1.GetGeoTransform()
        gt2 = ds_2.GetGeoTransform()

        # Get the image corners in pixel coordinates for the first image
        width1   = ds_1.RasterXSize
        height1  = ds_1.RasterYSize
        width2   = ds_2.RasterXSize
        height2  = ds_2.RasterYSize

        num_bands = ds_1.RasterCount
        if ds_2.RasterCount != num_bands:
            raise ValueError(f"Images have different number of bands: {num_bands} vs {ds_2.RasterCount}")

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
        
        # Calculate overlap bounds
        overlap_west  = max(min(corner_xs1), min(corner_xs2))
        overlap_east  = min(max(corner_xs1), max(corner_xs2))
        overlap_south = max(min(corner_ys1), min(corner_ys2))
        overlap_north = min(max(corner_ys1), max(corner_ys2))
        
        # Check if there is an overlap
        if overlap_east <= overlap_west or overlap_north <= overlap_south:
            return None, None
        
        # Create a grid for the overlapping region
        grid_size    = 500                                       # Tuning parameter, grid height (grid width is adjusted) (automatic grid size calculation?)
        width        = overlap_east - overlap_west
        height       = overlap_north - overlap_south
        aspect_ratio = width / height
        grid_width   = int(grid_size * aspect_ratio)
        grid_height  = grid_size
        
        # Create coordinate grids
        grid_x = np.linspace(overlap_west, overlap_east, grid_width)
        grid_y = np.linspace(overlap_south, overlap_north, grid_height)
        # Create meshgrid for pixel coordinates in the overlap region (world to pixel)
        X, Y   = np.meshgrid(grid_x, grid_y)

        # Get pixel coordinates of the overlap region for both images in local pixel coordinates
        pixel_x1, pixel_y1 = self._world_to_pixel(X, Y, gt1)
        pixel_x2, pixel_y2 = self._world_to_pixel(X, Y, gt2)

        # Create masks for valid pixels for image1 and image2
        mask1 = ((pixel_x1 >= 0) & (pixel_x1 < width1) & 
                 (pixel_y1 >= 0) & (pixel_y1 < height1))
        mask2 = ((pixel_x2 >= 0) & (pixel_x2 < width2) & 
                 (pixel_y2 >= 0) & (pixel_y2 < height2))

        # Combined mask for overlap
        overlap_mask = mask1 & mask2
        
        # Read the first band only for feature matching
        band1 = ds_1.GetRasterBand(1)
        band2 = ds_2.GetRasterBand(1)
        data1 = band1.ReadAsArray()
        data2 = band2.ReadAsArray()

        # Create output arrays
        output1 = np.zeros((num_bands, grid_height, grid_width))
        output2 = np.zeros((num_bands, grid_height, grid_width))

        for band_idx in range(num_bands):
            band1 = ds_1.GetRasterBand(band_idx + 1)  # GDAL bands are 1-based
            band2 = ds_2.GetRasterBand(band_idx + 1)
            data1 = band1.ReadAsArray()
            data2 = band2.ReadAsArray()

            # Sample the data using pixel coordinates # TODO: This is not resulting in the correct output IMAGE IS BLACK
            valid_y, valid_x = np.where(overlap_mask)
            for i, j in zip(valid_y, valid_x):
                px1, py1                = int(pixel_x1[i, j]), int(pixel_y1[i, j])
                px2, py2                = int(pixel_x2[i, j]), int(pixel_y2[i, j])
                output1[band_idx, i, j] = data1[py1, px1]
                output2[band_idx, i, j] = data2[py2, px2]

        # Mask areas outside overlap
        output1 = np.ma.masked_array(output1, np.repeat(~overlap_mask[np.newaxis, :, :], num_bands, axis=0))
        output2 = np.ma.masked_array(output2, np.repeat(~overlap_mask[np.newaxis, :, :], num_bands, axis=0))
        # Store georeference information as attributes
        output1.geotransform = (overlap_west, 
                                (overlap_east - overlap_west) / grid_width,  # X resolution
                                0,                                          # X skew
                                overlap_south,                              # Y origin (changed from overlap_north)
                                0,                                          # Y skew
                                (overlap_north - overlap_south) / grid_height)  # Y resolution (negative)
        output1.projection = ds_1.GetProjection()

        output2.geotransform = output1.geotransform  # Same geotransform for both
        output2.projection   = ds_2.GetProjection()
        
        if save_overlap:
            # Save the overlap as a new geotiff
            if save_path is None:
                # Use parent folder
                save_path = os.path.dirname(img_name1)

            base_name1 = os.path.splitext(os.path.basename(img_name1))[0]
            base_name2 = os.path.splitext(os.path.basename(img_name2))[0]

            overlap_path1 = os.path.join(save_path, f"{base_name1}_overlap1.tif")
            overlap_path2 = os.path.join(save_path, f"{base_name2}_overlap2.tif")

            self._save_overlap_geotiff(output1, overlap_path1)
            self._save_overlap_geotiff(output2, overlap_path2)

        return output1, output2

    @staticmethod
    def _world_to_pixel(x, y, geotransform):
        det     = geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4]
        pixel_x = (geotransform[5] * (x - geotransform[0]) - 
                  geotransform[2] * (y - geotransform[3])) / det
        pixel_y = (-geotransform[4] * (x - geotransform[0]) + 
                  geotransform[1] * (y - geotransform[3])) / det
        return pixel_x, pixel_y

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
    
#            if hasattr(geo_array, 'geotransform') and data.geotransform is not None:
#                out_ds.SetGeoTransform(data.geotransform)
    
#            if hasattr(geo_array, 'projection') and data.projection is not None:
#                out_ds.SetProjection(data.projection)

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