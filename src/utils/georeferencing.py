import numpy   as np
import pyvista as pv
import pyproj

from osgeo import gdal, osr

""" PRIVATE FUNCTIONS """
def _lla_to_ecef(lat, lon, alt):
    # Convert the GPS coordinates to WGS84, ECEF
    ecef    = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla     = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    return np.array([x, y, z])

def _calc_R_eb(roll, pitch, yaw):
    # Calculate the rotation matrix body -> ECEF from roll, pitch, yaw
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y @ R_x

def _dem_2_mesh(path_dem, model_path, config):
    """
    A function for converting a specified DEM to a 3D mesh model (*.vtk, *.ply or *.stl). Consequently, mesh should be thought of as 2.5D representation.
    :param path_dem: string
    path to dem for reading
    :param model_path: string
    path to where 3D mesh model is to be written.
    :return: Nothing
    """
    dem = pv.read(path_dem)

    # Input and output file paths
    output_xyz = model_path.split(sep = '.')[0] + '.xyz'                                                     #Oliver: output file path for the file model.xyz

    # No-data value
    #no_data_value = int(config['General']['nodataDEM'])  # Replace with your actual no-data value

    # Open the input raster dataset
    ds = gdal.Open(path_dem)

    if ds is None:
        print(f"Failed to open {path_dem}")
    else:
        # Read the first band (band index is 1)
        band          = ds.GetRasterBand(1)
#        no_data_value = band.GetNoDataValue()
        if band is None:
            print(f"Failed to open band 1 of {path_dem}")
        else:
            # Get the geotransform information to calculate coordinates
            # This step gets the geotransform information from the raster dataset, so that the position of the DEM in the real world can be determined.
            geotransform = ds.GetGeoTransform()
            x_origin     = geotransform[0]                                              #The x-coordinate of the upper-left corner of the top-left pixel.
            y_origin     = geotransform[3]                                              #The y-coordinate of the upper-left corner of the top-left pixel.
            x_resolution = geotransform[1]                                              #The pixel width in the x-direction.
            y_resolution = geotransform[5]                                              #The pixel height in the y-direction.
            # Get the CRS information
            spatial_reference = osr.SpatialReference(ds.GetProjection())

            # Get the EPSG code
            epsg_proj = None
            if spatial_reference.IsProjected():
                epsg_proj = spatial_reference.GetAttrValue("AUTHORITY", 1)              #Oliver: The EPSG code 32633
            elif spatial_reference.IsGeographic():
                epsg_proj = spatial_reference.GetAttrValue("AUTHORITY", 0)

            print(f"DEM projected EPSG Code: {epsg_proj}")

            config.set('Coordinate Reference Systems', 'dem_epsg', str(epsg_proj))
            
            # Get the band's data as a NumPy array
            band_data = band.ReadAsArray()
            # Create a mask to identify no-data values
            mask = band_data != no_data_value
            # Create and open the output XYZ file for writing if it does not exist:
            #if not os.path.exists(output_xyz):
            with open(output_xyz, 'w') as xyz_file:
                # Write data to the XYZ file using the mask and calculated coordinates
                for y in range(ds.RasterYSize):                                         #Oliver: This creates the model.xyz file, used for further processing.
                    for x in range(ds.RasterXSize):
                        if mask[y, x]:
                            x_coord = x_origin + x * x_resolution
                            y_coord = y_origin + y * y_resolution
                            xyz_file.write(f"{x_coord} {y_coord} {band_data[y, x]}\n")
            # Clean up
            ds   = None
            band = None
    print("Conversion completed.")
    points = np.loadtxt(output_xyz)                                                     #Oliver: Loads the model.xyz file into a numpy array (model.xyz was created a few lines ago).
    # Create a pyvista point cloud object
    cloud = pv.PolyData(points)
    # Generate a mesh from
    mesh = cloud.delaunay_2d()                                                          #Oliver: Creates a 2D mesh from the 3D point cloud.

    epsg_geocsc = config['Coordinate Reference Systems']['geocsc_epsg_export']
    # Transform the mesh points to from projected to geocentric ECEF.
    geocsc      = CRS.from_epsg(epsg_geocsc)
    proj        = CRS.from_epsg(epsg_proj)
    transformer = Transformer.from_crs(proj, geocsc)

    print(f"Mesh geocentric EPSG Code: {epsg_geocsc}")

    #Oliver: Projecting mesh from EPSG 32633 to EPSG 4326 (WGS84 geocentric/ECEF)
    points_proj = mesh.points

    eastUTM  = points_proj[:, 0].reshape((-1, 1))
    northUTM = points_proj[:, 1].reshape((-1, 1))
    heiUTM   = points_proj[:, 2].reshape((-1, 1))

    (xECEF, yECEF, zECEF) = transformer.transform(xx=eastUTM, yy=northUTM, zz=heiUTM)

    mesh.points[:, 0] = xECEF.reshape(-1)
    mesh.points[:, 1] = yECEF.reshape(-1)
    mesh.points[:, 2] = zECEF.reshape(-1)

    #mean_vec = np.mean(mesh.points, axis = 0)

    offX = float(config['General']['offsetX'])
    offY = float(config['General']['offsetY'])
    offZ = float(config['General']['offsetZ'])

    pos0 = np.array([offX, offY, offZ]).reshape((1, -1))

    mesh.points -= pos0 # Add appropriate offset
    # Save mesh
    mesh.save(model_path)


""" PUBLIC FUNCTIONS """
def calculate_camera_position(lat, lon, alt, roll, pitch, yaw, init_param):
    # Convert the GPS coordinates to WGS84, ECEF
    p_eb_e = _lla_to_ecef(lat, lon, alt)
    p_bc_b = init_param['p_bc_b']

    # Calculate the rotation matrix body -> ECEF from roll, pitch, yaw
    R_e_b = _calc_R_eb(roll, pitch, yaw)

    p_ec_e = p_eb_e + R_e_b @ p_bc_b
    return p_ec_e, R_e_b


def calculate_boresight_mesh_intersection(p_ec_e, R_e_b, init_param):
    # Calculate the boresight vector b_c_e in ECEF
    b_c_b = np.array([0, 0, 1])  # Boresight vector in body frame
    b_c_e = R_e_b @ b_c_b        # Boresight vector in ECEF

    # Load DEM mesh
    mesh = _dem_2_mesh(init_param['dem_path'], init_param['model_path'], init_param) #TODO: Oliver: This function shall be improved

    """Intersects the boresight-ray of the camera with the 3D triangular mesh

        :param mesh:           A mesh object read via the pyvista library
        :type mesh:            Pyvista mesh
        :param max_ray_length: The upper bound length of the camera rays (it is determined )
        :type max_ray_length:  _type_
    """


    n = self.rayDirectionsGlobal.shape[0]
    m = self.rayDirectionsGlobal.shape[1]

    self.points_ecef_crs  = np.zeros((n, m, 3), dtype=np.float64)
    self.normals_ecef_crs = np.zeros((n, m, 3), dtype=np.float64)



    # Calculate the intersection of the boresight vector with the ground


    pass
    
