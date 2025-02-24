import os
import numpy                as np
import tools.visualizations as vis

from tqdm             import tqdm
from osgeo            import gdal
from pathlib          import Path
from typing           import Dict
from scipy.signal     import correlate2d, fftconvolve
from scipy.fft        import fft2, ifft2
import cv2
from scipy.optimize import minimize
from scipy.signal import correlate2d

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

            # Extract the overlapping region between the current image and the previous image (n-1)
            #  If there is no overlap between the images, the function returns None, None
#            overlap_n_1, overlap_n = self._get_overlap_dataset(image_list[i-1]["filepath"],  # Base image (n-1)
#                                                               image_list[i]["filepath"],    # Current image (n)
#                                                               save_overlap=True,
#                                                               save_path="C:\\DocumentsLocal\\07_Code\\SeaBee\\SeaBee_georef_seagulls\\DATA\\")    
            overlap_n_1, overlap_n = self._get_overlap_dataset(image_list[i-1]["filepath"],  # Base image (n-1)
                                                               image_list[i]["filepath"],    # Current image (n)
                                                               save_overlap=False)

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
#            x_shift, y_shift = self._calculate_shift_conv(overlap_n_1, overlap_n)
            x_shift, y_shift = self._calculate_shift_manual(overlap_n_1, overlap_n, band=1)
            
            # Store the shift for the image
            shifts[img["filepath"]] = np.array([x_shift, y_shift])
            # Calculate the total shift
            total_shift = prev_shifts + np.array([x_shift, y_shift])
            prev_shifts = total_shift

            # Apply the shift to the image
            img_name = image_list[i]["filepath"]
            self._apply_shift(self.images[img_name], total_shift[0], total_shift[1])
            processed_set.add(i)

    def _get_overlap_dataset(self, img_name1, img_name2, save_overlap=False, save_path=None, no_data_value=np.nan):
        """
        Extracts the exact overlapping region between two images.
        
        INPUT: 
          img_name1:    Image name of the first image as string (same as key() values in self.images)
          img_name2:    Image name of the second image as string (same as key() values in self.images)
          save_overlap: Boolean to save the overlap as a new geotiff file (default: False)
          save_path:    Path to save the overlap geotiff files (default: None)
          
        OUTPUT: 
          Two overlapping datasets with identical bounds and dimensions.
          The datasets contain geotransform and projection information as attributes.

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
# Initialize output arrays with no_data_value
        output1 = np.full((num_bands, grid_height, grid_width), no_data_value, dtype=np.float32)
        output2 = np.full((num_bands, grid_height, grid_width), no_data_value, dtype=np.float32)


#        output1 = np.ma.masked_equal(output1, no_data_value)
#        output2 = np.ma.masked_equal(output2, no_data_value)

        for band_idx in range(num_bands):
            band1 = ds_1.GetRasterBand(band_idx + 1)  # GDAL bands are 1-based
            band2 = ds_2.GetRasterBand(band_idx + 1)
            # Read the band data
            data1 = band1.ReadAsArray()
            data2 = band2.ReadAsArray()

            # Get the no-data value for the band
            band1_no_data_value = band1.GetNoDataValue()
            band2_no_data_value = band2.GetNoDataValue()

            # Sample the data using pixel coordinates # TODO: This is not resulting in the correct output IMAGE IS BLACK
            valid_y, valid_x = np.where(overlap_mask)
            for i, j in zip(valid_y, valid_x):
                px1, py1                = int(pixel_x1[i, j]), int(pixel_y1[i, j])
                px2, py2                = int(pixel_x2[i, j]), int(pixel_y2[i, j])

                val1 = data1[py1, px1]
                val2 = data2[py2, px2]

                if band1_no_data_value is not None and val1 == band1_no_data_value:
                    output1[band_idx, i, j] = no_data_value
                else:
                    output1[band_idx, i, j] = val1

                if band2_no_data_value is not None and val2 == band2_no_data_value:
                    output2[band_idx, i, j] = no_data_value
                else:
                    output2[band_idx, i, j] = val2

        # Create masked arrays
        output1 = np.ma.masked_equal(output1, no_data_value)
        output2 = np.ma.masked_equal(output2, no_data_value)

        # Mask areas outside overlap
#        output1 = np.ma.masked_array(output1, np.repeat(~overlap_mask[np.newaxis, :, :], num_bands, axis=0))
#        output2 = np.ma.masked_array(output2, np.repeat(~overlap_mask[np.newaxis, :, :], num_bands, axis=0))

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
    
    def _calculate_shift_manual(self, ref_overlap_img, target_overlap_img, band=None):
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
            valid = (x_shifted >= 0) & (x_shifted < w) & (y_shifted >= 0) & (y_shifted < h)
            if not np.any(valid):
                return -1.0, 0

            # Sample shifted pixels
            x_sample = x_shifted[valid].astype(int)
            y_sample = y_shifted[valid].astype(int)

            # Get valid samples from both images
            ref_valid    = ref_img[y[valid], x[valid]]
            target_valid = target_img[y_sample, x_sample]

            # Remove any remaining NaNs
            valid_values = ~np.isnan(ref_valid) & ~np.isnan(target_valid)
            if not np.any(valid_values):
                return -1.0, 0

            ref_valid    = ref_valid[valid_values]
            target_valid = target_valid[valid_values]

            # Need minimum number of points for reliable correlation
            if len(ref_valid) < min_valid_pixels:
                return -1.0, 0

            # Compute NCC
            ref_mean    = np.mean(ref_valid)
            target_mean = np.mean(target_valid)

            ref_std    = np.std(ref_valid)
            target_std = np.std(target_valid)

            if ref_std < variance_threshold or target_std < variance_threshold:
                return -1.0, 0

            ref_norm    = (ref_valid - ref_mean) / ref_std
            target_norm = (target_valid - target_mean) / target_std

            ncc = np.mean(ref_norm * target_norm)

            return ncc, len(ref_valid)
        
        def grid_search(center_x, center_y, radius, step):
            """Perform grid search around a center point with given radius and step size."""
            best_ncc = float('-inf')
            best_shift = (center_x, center_y)
            best_valid_pixels = 0

            x_range = range(center_x - radius, center_x + radius + 1, step)
            y_range = range(center_y - radius, center_y + radius + 1, step)

            for x_shift in x_range:
                for y_shift in y_range:
                    ncc, valid_pixels = compute_ncc((x_shift, y_shift))
                    if ncc > best_ncc and valid_pixels >= min_valid_pixels:
                        best_ncc = ncc
                        best_shift = (x_shift, y_shift)
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

        ## DEBUGGING
        #import matplotlib.pyplot as plt
        #plt.figure(figsize=(10, 4))
        #plt.subplot(1, 2, 1)
        #plt.imshow(ref_overlap_img[band_idx, :, :].data, cmap='gray')
        #plt.title('Reference Image')
        #plt.subplot(1, 2, 2)
        #plt.imshow(target_overlap_img[band_idx, :, :].data, cmap='gray')
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

        #import matplotlib.pyplot as plt
        #plt.figure(figsize=(10, 4))
        #plt.subplot(1, 2, 1)
        #plt.imshow(ref_img, cmap='gray')
        #plt.title('Reference Image')
        #plt.subplot(1, 2, 2)
        #plt.imshow(target_img, cmap='gray')
        #plt.title('Target Image')
        #plt.show()

        # Print initial image statistics
        print(f"Reference image stats - min: {np.nanmin(ref_img):.4f}, max: {np.nanmax(ref_img):.4f}, mean: {np.nanmean(ref_img):.4f}")
        print(f"Target image stats - min: {np.nanmin(target_img):.4f}, max: {np.nanmax(target_img):.4f}, mean: {np.nanmean(target_img):.4f}")

        # Grid search parameters
        max_search_radius = 40  # Maximum shift to try in any direction
        min_valid_pixels = 100  # Minimum overlap required
        variance_threshold = 0.02  # Minimum variance required for reliable correlation
        ncc_threshold = 0.5  # Minimum NCC required for reliable correlation

        # First pass: Coarse search (step size = 4)
        print("\nCoarse grid search...")
        best_shift, best_ncc, best_valid_pixels = grid_search(0, 0, max_search_radius, 8)
    
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
        # NCC-based optimization
        # => Goal: Minimize the negative of the NCC (maximize the NCC)
        # => Use a negative sign to convert the maximization problem to a minimization problem
        # => Gradient-based optimization

    def _calculate_shift_conv(self, ref_overlap_img, overlap_img, method='auto', band=None):
        """
        Calculate shift between two images, optimized for handling large displacements.
        Returns shifts in whole pixels.
    
        Parameters:
        -----------
        ref_overlap_img : ndarray
            Reference image
        overlap_img : ndarray
            Image to align
        method : str
            'auto', 'fft', 'spatial' or 'corr2D'
        band : int, optional
            Specific band to use for matching
        
        Returns:
        --------
        ndarray
            [y_shift, x_shift] in pixels
        """
        # Input validation
        if ref_overlap_img.shape != overlap_img.shape:
            raise ValueError("Images must have the same shape")
    
        # Choose method based on image size if auto
        if method == 'auto':
            total_pixels = np.prod(ref_overlap_img.shape)
            method       = 'fft' if total_pixels > 1_000_000 else 'spatial'

        # Pre-calculate shapes
        ref_shape1, ref_shape2 = np.array(ref_overlap_img.shape[-2:]) - 1
    
        # Process each band
        shifts      = []
        confidences = []

        # Determine which bands to process
        if band is not None:
            bands_to_process = [band]
        else:
            bands_to_process = range(ref_overlap_img.shape[0])

        for b in bands_to_process:
            # Normalize images
            ref_norm     = self._normalize_image(ref_overlap_img[b])
            overlap_norm = self._normalize_image(overlap_img[b])

            # Add edge enhancement to improve feature matching
            edge_weight   = 0.3  # Reduced from 0.5 to avoid over-emphasizing noise
            ref_edges     = self._compute_edges(ref_norm)
            overlap_edges = self._compute_edges(overlap_norm)

            ref_enhanced     = ref_norm + edge_weight * ref_edges
            overlap_enhanced = overlap_norm + edge_weight * overlap_edges

            if method == 'fft':
                # Apply window function to reduce edge effects
                window = np.outer(np.hanning(ref_enhanced.shape[0]), 
                                np.hanning(ref_enhanced.shape[1]))
                ref_windowed = ref_enhanced * window
                overlap_windowed = overlap_enhanced * window

                # Compute correlation using FFT
                f1 = fft2(ref_windowed)
                f2 = fft2(overlap_windowed)
                correlation = np.real(ifft2(f1 * f2.conj()))
            elif method == 'spatial':
                # Use spatial correlation (this only works if the missalignment is purely translational (no rotation or scaling))
                correlation = fftconvolve(ref_enhanced, 
                                        overlap_enhanced[::-1, ::-1], 
                                        mode='full')
            elif method == 'corr2D':
                # Use correlation2d (this only works if the missalignment is purely translational (no rotation or scaling))
                correlation = correlate2d(ref_enhanced, overlap_enhanced, mode='full')
            else:
                raise ValueError("Invalid method. Use 'auto', 'fft', 'spatial', or 'corr2D'")

            # Find the peak in correlation
            y_max, x_max = np.unravel_index(np.argmax(correlation), correlation.shape)

            # Calculate shift in pixels
            shift = np.array([
                y_max - ref_shape1,
                x_max - ref_shape2
            ])

            # Calculate confidence based on correlation peak strength
            peak_val = np.max(correlation)
            mean_val = np.mean(correlation)
            std_val = np.std(correlation)
            confidence = (peak_val - mean_val) / std_val
    
            shifts.append(shift)
            confidences.append(confidence)
    
        # Convert to arrays
        shifts = np.array(shifts)
        confidences = np.array(confidences)
    
        # Filter out low confidence measurements
        min_confidence = 2.0  # Increased threshold for more reliable matches
        valid_shifts = shifts[confidences > min_confidence]
        valid_confidences = confidences[confidences > min_confidence]
    
        if len(valid_shifts) == 0:
            print("Warning: No reliable shifts found. Using best available match.")
            # Use the shift with highest confidence
            best_idx = np.argmax(confidences)
            final_shift = shifts[best_idx]
        else:
            # Use weighted average of valid shifts
            final_shift = np.average(valid_shifts, weights=valid_confidences, axis=0)
    
        # Round to whole pixels
        final_shift = np.round(final_shift).astype(int)
    
        # Add sanity check for unreasonably large shifts
        max_reasonable_shift = min(ref_shape1, ref_shape2) // 2
        if np.any(np.abs(final_shift) > max_reasonable_shift):
            print(f"Warning: Large shift detected {final_shift}. This might indicate a matching error.")
            # Clip to reasonable range
            final_shift = np.clip(final_shift, -max_reasonable_shift, max_reasonable_shift)
    
        return final_shift

    def _compute_edges(self, img):
        """
        Compute edge magnitude using Sobel operator.
        Simplified version focusing on strong edges.
        """
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        edges_x = correlate2d(img, sobel_x, mode='same', boundary='symm')
        edges_y = correlate2d(img, sobel_y, mode='same', boundary='symm')
        
        edges = np.sqrt(edges_x**2 + edges_y**2)
        
        # Normalize edge magnitudes
        edges = edges / (np.max(edges) + 1e-10)
        
        return edges

    @staticmethod
    def _normalize_image(img):
        img_norm = img.astype(float)
        img_norm -= np.mean(img_norm)
        std = np.std(img_norm)
        if std > 0:
            img_norm /= std
        return img_norm