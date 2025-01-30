import os, sys
from __init__     import initialize
import tools.visualizations as vis

#import numpy              as np
# matplotlib.pyplot  as plt
#import matplotlib.patches as patches

#from PIL          import Image
#from PIL.ExifTags import TAGS, GPSTAGS
from pyproj       import Transformer
#from osgeo        import gdal, osr

from objects.GeoPose import GeoPose
from objects.DEM     import DEM
from objects.FlatDEM import FlatDEM
from objects.DEM_J2  import DEM_J2

def main():
    geoPose = GeoPose(config, parameter)
    geoPose.calculate_camera_position()

    # Load the 3D mesh generated from the DEM (in EPGS 4326 / WGS84 coordinates)
    dem     = DEM(config) #['ELEVATION MODELS']['model_path']
    demFlat = FlatDEM(63.0, 65.0, 8.0, 10.0, 0.0, 1000) # DEBUGGING PURPOSES -------------> FIX THIS
    #J2_DEM  = DEM_J2(1.0, -1.0, 1.0, -1.0, 1000) # DEBUGGING PURPOSES        -------------> FIX THIS
    vis.visualize_mesh(dem.mesh, title='Mesh Visualization', xlabel='east', ylabel='north', zlabel='hight', coordinateSystem=None)
    vis.visualize_mesh(demFlat.mesh, title='Mesh Visualization', xlabel='east', ylabel='north', zlabel='hight', coordinateSystem=None)

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
