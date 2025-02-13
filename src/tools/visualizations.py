

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from matplotlib import pyplot as plt
import pyvista as pv
import pyproj as pypr
import tools.mesh_manipulation as mm

from rasterio.plot import plotting_extent, show
from rasterio.transform import Affine
from matplotlib.patches import Polygon
from rasterio.features import geometry_window



def visualize_mesh_and_camera_rays(mesh, geoPose, title='Mesh Visualization', coordinateSystem=None, xlabel='-', ylabel='-', zlabel='-', show_axes=False, buffer_around_p_ec_e=-1, show_camera_rays=False):
    """
    Visualize a mesh and camera rays:
    INPUTS:
    mesh:                 PyVista mesh object (the mesh to be visualized, generated from a DEM)
    geoPose:              GeoPose object (camera pose)
    title:                string (title of the plot)                                    [default: 'Mesh Visualization']
    coordinateSystem:     string (coordinate system)                                    [default: None] -> If None, no coordinate system is shown
    xlabel:               X-label of the coordinate system                              [default: '-']  -> If coordinateSystem == None, not used
    ylabel:               Y-label of the coordinate system                              [default: '-']  -> If coordinateSystem == None, not used
    zlabel:               Z-label of the coordinate system                              [default: '-']  -> If coordinateSystem == None, not used
    show_axes:            boolean (whether to show axes)                                [default: False]
    buffer_around_p_ec_e: float (buffer around the camera position in ECEF coordinates) [default: -1 => show the whole mesh]
    show_camera_rays:     boolean (whether to show camera rays)                         [default: False]
    """
    # Create a plotter
    plotter = pv.Plotter()

    # Show bounds
    if show_axes == True:
        plotter.show_bounds(
            grid='front',
            location='outer',
            ticks='outside',
            font_size=10,
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
        )

    # Add mesh with edges visible
    if buffer_around_p_ec_e > 0:
        mesh_cropped, bounds_ecef = mm.crop_mesh_by_bounds(mesh, geoPose, offset=buffer_around_p_ec_e)

        plotter.add_mesh(mesh_cropped, style='surface', show_edges=True)

        # Add points as spheres
        plotter.add_mesh(mesh_cropped.points, render_points_as_spheres=True, point_size=1, color='red')
    else:
        plotter.add_mesh(mesh, style='surface', show_edges=True)
        # Add points as spheres
        plotter.add_mesh(mesh.points, render_points_as_spheres=True, point_size=1, color='red')

    # Add camera positions as blue spheres
    plotter.add_mesh(geoPose.p_ec_e, render_points_as_spheres=True, point_size=5, color='blue')
    #plotter.add_mesh(geoPose.p_eb_e, render_points_as_spheres=True, point_size=2, color='green')

    # Add camera rays
    if geoPose.v_c_e is not None and show_camera_rays == True:
        for i in range(geoPose.p_ec_e.shape[0]):
            start = geoPose.p_eb_e[i,:]
            end   = start + geoPose.l_cd_max * geoPose.v_c_e[i,:]           # Scale the ray length
            for j in range(end.shape[0]):
                plotter.add_mesh(pv.Line(start, end[j,:]), color='green')


    # Add axes
    plotter.add_axes()
    plotter.add_text(title)

    if coordinateSystem is not None:
        # Add coordinate axes at the minimum bounding box point
        bounds    = mesh.bounds                        # (xmin, xmax, ymin, ymax, zmin, zmax)
        min_point = (bounds[0], bounds[2], bounds[4])  # (xmin, ymin, zmin)
        
        axes = plotter.add_axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
        # Set the origin of the axes to the minimum point
        axes.origin = min_point

    # Show the plot
    plotter.show()

def visualize_mesh(mesh, title='Mesh Visualization', xlabel='x', ylabel='y', zlabel='z', coordinateSystem=None):
    """
    Visualize a mesh
    """
    # Create a plotter
    plotter = pv.Plotter()
    # Add mesh with edges visible
    plotter.add_mesh(mesh, style='surface', show_edges=True)
    # Add points as spheres
    plotter.add_mesh(mesh.points, render_points_as_spheres=True, 
                    point_size=10, color='red')
    # Add axes
    plotter.add_axes()
    # Add title
    plotter.add_text(title)
    # Add coordinate axes
#    plotter.add_axes_labels(xlabel, ylabel, zlabel)
    if coordinateSystem is not None:
        plotter.add_axes_at_origin(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    # Show the plot
    plotter.show()

def visualize_pointcloud(pointcloud, subsample_factor=1):
    """ Visualize point cloud
    """
    # Create a figure and axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Display the point cloud
    x = pointcloud[:, 0][::subsample_factor]
    y = pointcloud[:, 1][::subsample_factor]
    z = pointcloud[:, 2][::subsample_factor]
    ax.scatter(x, y, z, c=z, cmap='viridis')
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Add title
    ax.set_title('Point Cloud Visualization')
    # Display the plot
    plt.show()


def visualize_dem(band):
    """ Visualize DEM / DTM / DOM
    """
    rows, cols = band.shape
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    x, y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, band, cmap='terrain', linewidth=0, antialiased=True)
    ax.set_xlabel('Northing')
    ax.set_ylabel('Easting')
    ax.set_zlabel('Elevation')
    fig.colorbar(surf, ax=ax, label='Elevation')
    plt.title('DEM Visualization')
    plt.show()

def visualize_mosaik(images, first_idx, second_idx):
    """
    Visualize a selection of georeferenced images as a photomosaic in their correct spatial positions.
    
    Parameters:
    images (list): List of image dictionaries containing img_data, transform, and other metadata
    first_idx (int): Starting index for the image selection
    second_idx (int): Ending index for the image selection
    """
    axes_set = False

    fig, ax = plt.subplots(figsize=(15, 10))

    # Process selected images
    for i in range(first_idx, second_idx + 1):
        if i >= len(images):
            break

        # Get transform and image dimensions
        rastImg      = images[i]["rastImg"]
        transform    = rastImg.transform
        height       = rastImg.height
        width        = rastImg.width

        # Calculate corner coordinates in world space
        corners = [
            transform * (0, 0),          # Upper-left
            transform * (width, 0),      # Upper-right
            transform * (width, height), # Lower-right
            transform * (0, height),     # Lower-left
        ]
        extent=plotting_extent(rastImg)

        show(rastImg.read(), ax=ax, transform=transform)

        # Add outline to show image boundaries
        poly = Polygon(corners, closed=True, edgecolor='r', facecolor='none', alpha=0.5, linewidth=0.5)
        ax.add_patch(poly)

        # Set labels if CRS info is available
        if "EPSG" in rastImg.crs.wkt and not axes_set:
            axes_set = True
            if "UTM" in rastImg.crs.wkt:
                # Extract UTM zone number as "UTM zone XX"
                utm_name = "UTM Zone " + rastImg.crs.wkt.split("UTM zone ")[1].split(",")[0]
                ax.set_xlabel("Easting (m)")
                ax.set_ylabel("Northing (m)")
            elif "WGS 84" in rastImg.crs.wkt:
                # TODO improve
                ax.set_xlabel("X Coordinate")
                ax.set_ylabel("Y Coordinate")

    # Auto-adjust view and add title
    ax.autoscale()
    plt.title(f"Georeferenced Image Mosaic in coordinates {utm_name} (Images {first_idx} to {second_idx})")
    plt.tight_layout()
    plt.show()
