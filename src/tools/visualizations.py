

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from matplotlib import pyplot as plt
import pyvista as pv
import pyproj as pypr
import tools.mesh_manipulation as mm

def visualize_mesh_and_camera_rays(mesh, geoPose, title='Mesh Visualization', xlabel='east', ylabel='north', zlabel='hight', coordinateSystem=None, limit_area=False):
    """
    Visualize a mesh and camera rays
    """
    # Create a plotter
    plotter = pv.Plotter()

    # Add mesh with edges visible
    if limit_area == True:
        mesh_cropped, bounds_ecef = mm.crop_mesh_by_bounds(mesh, geoPose)
#        mesh_cropped = mm.crop_mesh_by_bounds_simple(mesh, geoPose.p_ec_e)
        plotter.add_mesh(mesh, style='surface', show_edges=True)

        # Add points as spheres
        plotter.add_mesh(mesh.points, render_points_as_spheres=True, point_size=1, color='red')
        plotter.add_mesh(bounds_ecef, color='blue', opacity=0.5)
    else:
        plotter.add_mesh(mesh, style='surface', show_edges=True)

        # Add points as spheres
        plotter.add_mesh(mesh.points, render_points_as_spheres=True, point_size=1, color='red')

    # Add camera positions as blue spheres
    plotter.add_mesh(geoPose.p_ec_e, render_points_as_spheres=True, point_size=2, color='blue')
    #plotter.add_mesh(geoPose.p_eb_e, render_points_as_spheres=True, point_size=2, color='green')

    # Add camera rays
    if geoPose.v_c_e is not None:
        for i in range(geoPose.p_ec_e.shape[0]):
            start = geoPose.p_eb_e[i,:]
            end   = start + geoPose.l_cd_max * geoPose.v_c_e[i,:]  # Scale the ray length
            plotter.add_mesh(pv.Line(start, end), color='green')

    # Add axes
    plotter.add_axes()
    plotter.add_text(title)

    if coordinateSystem is not None:
        plotter.add_axes_at_origin(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    # Show the plot
    plotter.show()

def visualize_mesh(mesh, title='Mesh Visualization', xlabel='east', ylabel='north', zlabel='hight', coordinateSystem=None):
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

