import os
import tools.visualizations as vis

from __init__                    import initialize
from objects.GeoPose             import GeoPose
from objects.DEM                 import DEM
from objects.DEM_J2              import DEM_J2
from objects.Georectification    import Georectification
from objects.Orthorectification  import Orthorectification

def main():
    geoPose = GeoPose(config, parameter)
    # Potential Improvement: Kalman Filter to improve the GeoPose accuracy
    geoPose.calculate_camera_position()

    # Check if a custom DEM is available
    if os.path.exists(config['ELEVATION MODELS']['dem_path']):
        print('Custom DEM available!')
        # Load the 3D mesh generated from the DEM (in EPGS 4326 / WGS84 coordinates)
        dem     = DEM(config) #['ELEVATION MODELS']['model_path']
    else:
        # Generate a DEM based on WGS-84 ellipsoid with J2 perturbation
        dem  = DEM_J2(1.0, -1.0, 1.0, -1.0, 1000) # In case no custom DEM is available, use the J2 DEM
#    vis.visualize_mesh(dem.mesh, title='Mesh Visualization', xlabel='east', ylabel='north', zlabel='hight', coordinateSystem=None)

    # Intersection of camera center axis with ground DEM
    geoPose.boresight_mesh_intersection(dem)
#    vis.visualize_mesh_and_camera_rays(dem.mesh, geoPose)
#    vis.visualize_mesh_and_camera_rays(dem.mesh, geoPose, title='Mesh Visualization', coordinateSystem='ECEF', xlabel='X', ylabel='Y', zlabel='Z', show_axes=True, buffer_around_p_ec_e=1000, show_camera_rays=True)

    ortho = Georectification(config, parameter, dem=dem, geoPose=geoPose)
    ortho.georectify_all() # Georeference images and convert to EPSG-25833
    print('Georectification done!')

    if config['ORTHORECTIFICATION']['orthorectify_images'] == 'True':
        print("Start Orthorectification")
        featureMatch = Orthorectification(config)
        featureMatch.main_orthorectify_images()

if __name__ == '__main__':
    config, parameter = initialize()
    main()
