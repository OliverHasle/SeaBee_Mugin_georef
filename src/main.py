import os, sys
from __init__     import initialize
import tools.constants       as c
import tools.visualizations  as vis

#import matplotlib.patches as patches

#from PIL          import Image
#from PIL.ExifTags import TAGS, GPSTAGS
#from osgeo        import gdal, osr

from pyproj                     import Transformer
from objects.GeoPose            import GeoPose
from objects.DEM                import DEM
from objects.Orthorectification import Orthorectification
from objects.DEM_J2             import DEM_J2

def main():
    geoPose = GeoPose(config, parameter)
    geoPose.calculate_camera_position()

    # Load the 3D mesh generated from the DEM (in EPGS 4326 / WGS84 coordinates)
    dem     = DEM(config) #['ELEVATION MODELS']['model_path']
    #J2_DEM  = DEM_J2(1.0, -1.0, 1.0, -1.0, 1000) # DEBUGGING PURPOSES        -------------> FIX THIS
#    vis.visualize_mesh(dem.mesh, title='Mesh Visualization', xlabel='east', ylabel='north', zlabel='hight', coordinateSystem=None)

    # Intersection of camera center axis with ground DEM
    geoPose.boresight_mesh_intersection(dem)
#    vis.visualize_mesh_and_camera_rays(dem.mesh, geoPose)
#    vis.visualize_mesh_and_camera_rays(dem.mesh, geoPose, title='Mesh Visualization', coordinateSystem='ECEF', xlabel='X', ylabel='Y', zlabel='Z', show_axes=True, buffer_around_p_ec_e=1000, show_camera_rays=True)

    ortho = Orthorectification(config, parameter, geoPose=geoPose, dem=dem)
    ortho.georectify_all() # Georeference images and convert to EPSG-25833
    print('Orthorectification done!')

if __name__ == '__main__':
    config, parameter = initialize()
    main()
