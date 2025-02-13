import os
from tqdm             import tqdm
#from pyproj           import Transformer
from osgeo            import gdal, osr
from pathlib          import Path
from scipy.signal     import correlate2d
import tools.visualizations as vis

import rasterio    as rasterio
import numpy       as np
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

    @staticmethod
    def _find_overlap_bounds(bounds1, bounds2):
        """
        A function for finding the overlapping bounds of the images.
        """
        x_min = max(bounds1.left,   bounds2.left)
        y_min = max(bounds1.right,  bounds2.right)
        x_max = min(bounds1.bottom, bounds2.bottom)
        y_max = min(bounds1.top,    bounds2.top)
        return x_min, y_min, x_max, y_max
    
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

    def process_image_grid(self):
        """Process images from NW to SE, calculating required offsets."""
        offsets   = {}
        processed = set()
    
        # Convert dictionary values to a list for sorting
        image_list = list(self.images.values())
        
        # Sort images from North-West to South-East using the upper-left corner (top, left) coordinates
        # THIS IS NOT WORKING => RANDOMLY SORTED
        sorted_img_nw_se = sorted(image_list, 
                                  key=lambda img: (-img["rastImg"].bounds.top, img["rastImg"].bounds.left))
        
        vis.visualize_mosaik(sorted_img_nw_se, first_idx=0, second_idx=3)

        for i, img in enumerate(sorted_img_nw_se):
            path = img["filepath"]
            if i == 0:
                processed.add(path)
                continue

            img_data      = sorted_img_nw_se[i]["bands"][0]   # Using the first band
            img_transform = sorted_img_nw_se[i]["transform"]
            img_bounds    = sorted_img_nw_se[i]["bounds"]

            # Find neighboring processed images
            for processed_path in processed:
                ref_img       = sorted_img_nw_se[i]
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

    def tiels_matching(self):
        """
        A function for matching features between images.
        """
        pass

