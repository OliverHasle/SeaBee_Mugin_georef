import os, sys
from __init__     import initialize

#import numpy              as np
# matplotlib.pyplot  as plt
#import matplotlib.patches as patches

#from PIL          import Image
#from PIL.ExifTags import TAGS, GPSTAGS
from pyproj       import Transformer
#from osgeo        import gdal, osr

from objects.GeoPose import GeoPose
from objects.DEM     import DEM

def main():
    geoPose = GeoPose(config, parameter)
    geoPose.calculate_camera_position()

    # Load the 3D mesh generated from the DEM (in EPGS 4326 / WGS84 coordinates)
    dem = DEM(config) #['ELEVATION MODELS']['model_path']
    #dem.visualize_mesh(dem.mesh, title='Mesh Visualization', xlabel='east', ylabel='north', zlabel='hight', coordinateSystem=None)

    # Intersection of camera center axis with ground DEM
    geoPose.boresight_mesh_intersection(dem)

    # Set up transformer for UTM to WGS84 conversion
    utm_to_wgs84 = Transformer.from_crs("epsg:32632", "epsg:4326", always_xy=True)  # Replace 32632 with your UTM zone

    

    # Get image center position and rotation matrix   #center_x, center_y, rotation_matrix = 




    # get the list of images
    images = os.listdir(config['MISSION']['inputfolder'])
    a = 0

    # get the list of images

if __name__ == '__main__':
    config, parameter = initialize()
    main()
