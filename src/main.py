import os, sys
from __init__     import initialize

import numpy              as np
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import utils.optical_calc   as oc
import utils.georeferencing as georef

from PIL          import Image
from PIL.ExifTags import TAGS, GPSTAGS
#from libxmp       import XMPFiles
from pyproj       import Transformer
from osgeo        import gdal, osr

def main():
    fov_x = oc.calc_fov(init_param['sensor']['focal_length'], init_param['sensor']['sensor_width'])
    fov_y = oc.calc_fov(init_param['sensor']['focal_length'], init_param['sensor']['sensor_height'])

    # Open gpslog file
    gpslogfile = [filename for filename in os.listdir(init_param['inputfolder']) if filename.startswith("gpslog")]
    gpslogpath = os.path.join(init_param['inputfolder'], gpslogfile[0])  # Use the first gpslog file

    # Set up transformer for UTM to WGS84 conversion
    utm_to_wgs84 = Transformer.from_crs("epsg:32632", "epsg:4326", always_xy=True)  # Replace 32632 with your UTM zone

    # Initialize an empty list to store the GPS data
    gpslog = []

    # Open and read the GPS log line by line
    with open(gpslogpath, 'r') as file:
        for line in file:
            values    = line.strip().split(',')
            gps_entry = {
                "filename":  values[0],
                "latitude":  float(values[1]),
                "longitude": float(values[2]),
                "altitude":  float(values[3]),
                "roll":      float(values[4]),
                "pitch":     float(values[5]),
                "yaw":       float(values[6])
            }
            gpslog.append(gps_entry)

    # Example usage for each image entry
    for entry in gpslog:
#        filename  = entry["filename"]
        latitude  = entry["latitude"]
        longitude = entry["longitude"]
        altitude  = entry["altitude"]
        roll      = entry["roll"]
        pitch     = entry["pitch"]
        yaw       = entry["yaw"]

        # Get image center position and rotation matrix   #center_x, center_y, rotation_matrix = 
        p_ec_e, R_e_b = georef.calculate_camera_position(
            latitude,  longitude,   altitude, 
            roll,      pitch,       yaw,
            init_param)

        # Intersection of camera center axis with ground DEM
        georef.calculate_boresight_mesh_intersection(p_ec_e, R_e_b, init_param)


    # get the list of images
    images = os.listdir(init_param['inputfolder'])
    a = 0

    # get the list of images

if __name__ == '__main__':
    init_param, config = initialize()
    main()
