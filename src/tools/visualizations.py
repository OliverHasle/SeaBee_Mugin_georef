

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from matplotlib import pyplot as plt
import pyvista as pv
import pyproj as pypr
import tools.mesh_manipulation as mm

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
#        plotter.add_mesh(mesh_cropped, style='surface', show_edges=True, scalar_bar_args={'title':      'Elevation', 
#                                                                                          'vertical':   True,
#                                                                                          'position_x': 0.85,  # Position the bar on the right side
#                                                                                          'position_y': 0.05,  # Position from bottom
#                                                                                          'width':       0.1,
#                                                                                          'height':      0.7,
#            })

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

