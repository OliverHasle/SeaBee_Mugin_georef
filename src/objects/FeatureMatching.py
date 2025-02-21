import os
import numpy                as np
import tools.visualizations as vis

from tqdm             import tqdm
from osgeo            import gdal
from pathlib          import Path
from typing           import Dict
from scipy.signal     import correlate2d
from scipy.signal     import correlate2d, fftconvolve
from scipy.fft        import fft2, ifft2

class FeatureMatching:
    def __init__(self, config):
        self.config    = config
        self.images: Dict[str, ImageInfo] = {} # Image dictionary containing all the image information
        self.offsets   = None

        self.load_geotiffs()

    def load_geotiffs(self) -> None:
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
            x_shift, y_shift = self._calculate_shift(overlap_n_1, overlap_n)
            
            # Store the shift for the image
            shifts[img["filepath"]] = np.array([x_shift, y_shift])
            # Calculate the total shift
            total_shift = prev_shifts + np.array([x_shift, y_shift])
            prev_shifts = total_shift

            # Apply the shift to the image
            img_name = image_list[i]["filepath"]
            self._apply_shift(self.images[img_name], total_shift[0], total_shift[1])
            processed_set.add(i)

    def _get_overlap_dataset(self, img_name1, img_name2, save_overlap=False, save_path=None):
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
        self._save_image(img_name, gdal_img=ds)
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
        rows = []
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
    
    def _calculate_shift(self, ref_overlap_img, overlap_img, method='auto', band=None):
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
            'auto', 'fft', or 'spatial'
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
            method = 'fft' if total_pixels > 1_000 else 'spatial'
    
        # Pre-calculate shapes
        ref_shape1, ref_shape2 = np.array(ref_overlap_img.shape[-2:]) - 1
    
        # Process each band
        shifts = []
        confidences = []

        # Determine which bands to process
        if band is not None:
            bands_to_process = [band]
        else:
            bands_to_process = range(ref_overlap_img.shape[0])
    
        for b in bands_to_process:
            # Normalize images
            ref_norm = self._normalize_image(ref_overlap_img[b])
            overlap_norm = self._normalize_image(overlap_img[b])
    
            # Add edge enhancement to improve feature matching
            edge_weight = 0.3  # Reduced from 0.5 to avoid over-emphasizing noise
            ref_edges = self._compute_edges(ref_norm)
            overlap_edges = self._compute_edges(overlap_norm)
            
            ref_enhanced = ref_norm + edge_weight * ref_edges
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
            else:
                # Use spatial correlation
                correlation = fftconvolve(ref_enhanced, 
                                        overlap_enhanced[::-1, ::-1], 
                                        mode='full')

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
    def _calculate_shift2(self, ref_overlap_img, overlap_img, method='auto', band=None):
        """
        Calculate the shift between two images using either FFT-based or spatial correlation based on the 
        image overlap region. The shift is calculated in pixels.

        Parameters:
        -----------
        ref_overlap_img : ndarray
            Reference image
        overlap_img : ndarray
            overlap_img to align
        method : str
            'auto', 'fft', or 'spatial'
        
        Returns:
        --------
        ndarray
            [y_shift, x_shift]
        """
        # Input validation (Shapes of the overlaps must be identical)
        if ref_overlap_img.shape != overlap_img.shape:
            raise ValueError("Images must have the same shape")

        # Choose method based on image size if auto
        if method == 'auto':
            total_pixels = np.prod(ref_overlap_img.shape)
            method       = 'fft' if total_pixels > 1_000_000 else 'spatial'

        # Pre-calculate shapes
        ref_shape1, ref_shape2 = np.array(ref_overlap_img.shape[-2:]) - 1

        if band is not None:
            # Use one band to find the correct shift
            # Initialize shift array
            shift_ = np.zeros((1, 2))
            ref_norm     = self._normalize_image(ref_overlap_img[band])
            overlap_norm = self._normalize_image(overlap_img[band])

            # Calculate shift for a single band
            if method == 'fft':
                # FFT-based correlation
                window           = np.outer(np.hanning(ref_overlap_img.shape[1]), 
                                            np.hanning(ref_overlap_img.shape[2]))
                ref_windowed     = ref_norm * window
                overlap_windowed = overlap_norm * window
            
                # Compute correlation using FFT
                f1          = fft2(ref_windowed)
                f2          = fft2(overlap_windowed)
                correlation = np.real(ifft2(f1 * f2.conj()))
            elif method == 'spatial':
                # Spatial correlation using fftconvolve (faster than correlate2d)
                correlation = fftconvolve(ref_norm, 
                                          overlap_norm[::-1, ::-1], 
                                          mode='full')
            else:
                # correlation2d
                correlation = correlate2d(ref_norm, overlap_norm, mode='full')

            # Find peak using numba-accelerated function
            y_max, x_max = np.unravel_index(np.argmax(correlation), correlation.shape)
            return np.array([y_max - ref_shape1, x_max - ref_shape2])
        else:
            # Use all color bands to find the correct shift
            shifts        = []
            valid_weights = []
#            shift_ = np.zeros((ref_overlap_img.shape[0], 2))
            for band in range(ref_overlap_img.shape[0]):
                ref_norm     = self._normalize_image(ref_overlap_img[band])
                overlap_norm = self._normalize_image(overlap_img[band])
                if method == 'fft':
                    # FFT-based correlation
                    window           = np.outer(np.hanning(ref_overlap_img.shape[1]), 
                                                np.hanning(ref_overlap_img.shape[2]))
                    ref_windowed     = ref_norm * window
                    overlap_windowed = overlap_norm * window
                    f1               = fft2(ref_windowed)
                    f2               = fft2(overlap_windowed)
                    correlation      = np.real(ifft2(f1 * f2.conj()))
                elif method == 'spatial':
                    # Spatial correlation using fftconvolve (faster than correlate2d)
                    correlation = fftconvolve(ref_norm, 
                                              overlap_norm[::-1, ::-1], 
                                              mode='full')
                else:
                    # correlation2d
                    correlation = correlate2d(ref_overlap_img[band], overlap_img[band], mode='full')

                # Find peak using numba-accelerated function
                y_max, x_max = np.unravel_index(np.argmax(correlation), correlation.shape)
                shift = np.array([y_max - ref_shape1, x_max - ref_shape2])

                weight = np.max(correlation)

                if weight > 0.1:  # Threshold can be adjusted
                    shifts.append(shift)
                    valid_weights.append(weight)

            if not shifts:
                # If no valid shifts found, return zero shift
                return np.array([0, 0])

            shifts = np.array(shifts)
            valid_weights = np.array(valid_weights)

            # Calculate weighted average of shifts
            weighted_shifts = np.average(shifts, axis=0, weights=valid_weights)
            return weighted_shifts
    @staticmethod
    def _normalize_image(img):
        img_norm = img.astype(float)
        img_norm -= np.mean(img_norm)
        std = np.std(img_norm)
        if std > 0:
            img_norm /= std
        return img_norm