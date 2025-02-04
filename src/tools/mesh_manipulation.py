import numpy   as np
import pyvista as pv

def crop_mesh_by_bounds(mesh, geoPose, offset=10.0):
    """
    Crop a PyVista mesh using ECEF boundary points.
    
    Parameters:
    mesh (pyvista.PolyData): Input mesh to be cropped
    bounds_ecef (np.ndarray): Nx3 array of boundary points in ECEF coordinates
    
    Returns:
    pyvista.PolyData: Cropped mesh
    """
    #delta_lat, delta_lon = _calc_delta_lat_lon(geoPose.latitude, geoPose.longitude, horizontal_offset)
    #mesh_bounds = mesh.bounds
    #z_min, z_max = mesh_bounds[4], mesh_bounds[5]

    # Get maximal and minimal x, y, z values of the camera position p_ec_e
    x_min, x_max = np.min(geoPose.p_ec_e[:, 0]), np.max(geoPose.p_ec_e[:, 0])
    y_min, y_max = np.min(geoPose.p_ec_e[:, 1]), np.max(geoPose.p_ec_e[:, 1])
    z_min, z_max = np.min(geoPose.p_ec_e[:, 2]), np.max(geoPose.p_ec_e[:, 2])

    # Calculate bounding box with offset
    p_ec_e_bounding_box = np.array([
        [x_min - offset, y_min - offset, z_min - offset],
        [x_max + offset, y_min - offset, z_min - offset],
        [x_max + offset, y_max + offset, z_min - offset],
        [x_min - offset, y_max + offset, z_min - offset],
        [x_min - offset, y_min - offset, z_max + offset],
        [x_max + offset, y_min - offset, z_max + offset],
        [x_max + offset, y_max + offset, z_max + offset],
        [x_min - offset, y_max + offset, z_max + offset]
    ])

    # Show the area to be cropped in a plot (DEBUG)
    #p = pv.Plotter()
    #p.add_mesh(mesh, color='red')
    #p.add_points(p_ec_e_bounding_box, color='blue')
    #p.show()

    # Create clipping surface
    clip_surface = create_boundary_surface(p_ec_e_bounding_box)

    # Perform the clipping
    try:
        mesh_cropped = mesh.clip_surface(clip_surface, invert=True)
    except RuntimeError as e:
        print(f"Warning: Initial clipping failed, trying alternative method: {e}")
        # Alternative approach using boolean operations
        mesh_cropped = mesh.boolean_intersection(clip_surface)

    #compare_meshes(mesh_cropped, mesh)
    return mesh_cropped, p_ec_e_bounding_box

def create_boundary_surface(bounds_ecef):
    """
    Create a PyVista surface from ECEF boundary points that can be used for clipping.

    Parameters:
    bounds_ecef (np.ndarray): Nx3 array of boundary points in ECEF coordinates

    Returns:
    pyvista.PolyData: Surface that can be used for clipping
    """
    num_points = len(bounds_ecef) // 2

    faces = []
    for i in range(num_points):
        j = (i + 1) % num_points
        # Bottom triangle
        faces.extend([3, i, j, i + num_points])
        # Top triangle
        faces.extend([3, j, j + num_points, i + num_points])
    
    # Create top face
    for i in range(num_points - 2):
        faces.extend([3, num_points, num_points + i + 1, num_points + i + 2])
    
    # Create bottom face
    for i in range(num_points - 2):
        faces.extend([3, 0, i + 1, i + 2])
    
    # Create the surface
    surface = pv.PolyData(bounds_ecef, faces=np.array(faces))
    
    return clean_mesh(surface)

def clean_mesh(mesh):
    """
    Clean and repair a mesh to ensure it's suitable for boolean operations.
    
    Parameters:
    mesh (pyvista.PolyData): Input mesh
    
    Returns:
    pyvista.PolyData: Cleaned mesh
    """
    # Remove duplicate points
    mesh = mesh.clean(tolerance=1e-5)
    
    # Ensure mesh is triangulated
    mesh = mesh.triangulate()
    
    # Fill holes
    mesh = mesh.fill_holes(hole_size=mesh.length / 50)
    
    # Ensure consistent triangle orientations
    mesh = mesh.compute_normals(inplace=False)
    
    return mesh

def compare_meshes(mesh_cropped, mesh):
    """
    Visualize the cropping results with debug information.
    
    Parameters:
    mesh_cropped (pyvista.PolyData): Cropped mesh
    debug_info (dict): Debug information from crop_mesh_by_bounds
    """
    p = pv.Plotter(shape=(1, 2))
    
    # Original mesh with box
    p.subplot(0, 0)
    p.add_title('Original Mesh with Crop Box')
    p.add_mesh(mesh, color='red')
    
    # Cropped result
    p.subplot(0, 1)
    p.add_title('Cropped Result')
    p.add_mesh(mesh_cropped, color='lightblue')
    
    p.link_views()
    p.show()
