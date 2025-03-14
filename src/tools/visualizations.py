

import os
import numpy                   as np
import plotly.graph_objects    as go
import pyvista                 as pv
import tools.mesh_manipulation as mm

from osgeo              import gdal
from matplotlib         import pyplot as plt
from matplotlib.colors  import LinearSegmentedColormap
from matplotlib.path    import Path
from matplotlib.patches import PathPatch

def visualize_mesh_and_camera_rays(mesh, geoPose, title='Mesh Visualization', coordinateSystem=None, xlabel='-', ylabel='-', zlabel='-', show_axes=False, buffer_around_p_ec_e=-1, show_camera_rays=False) -> None:
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
            grid       = 'front',
            location   = 'outer',
            ticks      = 'outside',
            font_size  = 10,
            show_xaxis = True,
            show_yaxis = True,
            show_zaxis = True,
            xlabel     = xlabel,
            ylabel     = ylabel,
            zlabel     = zlabel,
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
        bounds      = mesh.bounds                        # (xmin, xmax, ymin, ymax, zmin, zmax)
        min_point   = (bounds[0], bounds[2], bounds[4])  # (xmin, ymin, zmin)
        axes        = plotter.add_axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
        # Set the origin of the axes to the minimum point
        axes.origin = min_point

    # Show the plot
    plotter.show()

def visualize_mesh(mesh, title='Mesh Visualization', xlabel='x', ylabel='y', zlabel='z', coordinateSystem=None) -> None:
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

def visualize_pointcloud(pointcloud, subsample_factor=1) -> None:
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

def visualize_dem(band)                                  -> None:
    """ Visualize DEM / DTM / DOM
    """
    rows, cols = band.shape
    x          = np.arange(0, cols)
    y          = np.arange(0, rows)
    x, y       = np.meshgrid(x, y)
    fig        = plt.figure(figsize=(12, 8))
    ax         = fig.add_subplot(111, projection='3d')
    surf       = ax.plot_surface(x, y, band, cmap='terrain', linewidth=0, antialiased=True)
    ax.set_xlabel('Northing')
    ax.set_ylabel('Easting')
    ax.set_zlabel('Elevation')
    fig.colorbar(surf, ax=ax, label='Elevation')
    plt.title('DEM Visualization')
    plt.show()

def plot_georeferenced_images(image_list, first_idx=0, last_idx=None, figsize=(12, 8), 
                              title=None, cmap='viridis', vmin=None, vmax=None, 
                              show_overlap=True, edge_width=0.05, grid_size=500, show_north_arrow=True, show_scale_bar=True) -> None:
    """
    Plot multiple georeferenced images in their original orientation using GDAL
    
    Parameters:
    -----------
    image_list : list
        List of dictionaries containing raster images with 'gdalImg' key containing path or open rasterio dataset
    first_idx : int
        Starting index for images to plot
    last_idx : int, optional
        Ending index for images to plot (exclusive)
    figsize : tuple
        Figure size (width, height) in inches
    title : str, optional
        Title for the plot
    cmap : str, optional
        Colormap to use for the raster
    vmin, vmax : float, optional
        Min and max values for color scaling
    show_overlap : bool, optional
        Whether to highlight areas where images overlap
    edge_width : float, optional
        Fraction of the image width to add as a buffer around the images
    grid_size : int, optional
        Number of grid cells along the longest dimension for overlap grid
    show_north_arrow : bool, optional
        Whether to show a north arrow on the plot
    show_scale_bar : bool, optional
        Whether to show a scale bar on the plot
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    if last_idx is None:
        last_idx = first_idx + 1

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate overall bounds across all images
    all_west  = float('inf')
    all_east  = float('-inf')
    all_south = float('inf')
    all_north = float('-inf')

    world_corners_list = []

    # First pass: compute bounds and get image properties
    for i in range(first_idx, last_idx):
        # Get the file path from the rasterio dataset if needed
        if os.path.exists(image_list[i]["filepath"]):
            file_path = image_list[i]["filepath"]
        else:
            print(f"WARNING: File {image_list[i]['filepath']} does not exist")
            continue
        # Open with GDAL
        ds = gdal.Open(file_path)
        if not ds:
            print(f"WARNING: Could not open {file_path}")
            continue

        # Get geotransform (gt)
        gt = ds.GetGeoTransform()

        # Calculate corner coordinates
        width  = ds.RasterXSize
        height = ds.RasterYSize

        # Get all four corners
        corners = [
            (0,     0),              # Upper left
            (width, 0),              # Upper right
            (width, height),         # Lower right
            (0,     height)          # Lower left
        ]

        # Transform corners to world coordinates
        world_corners = []
        for x, y in corners:
            world_x = gt[0] + x * gt[1] + y * gt[2]
            world_y = gt[3] + x * gt[4] + y * gt[5]
            world_corners.append((world_x, world_y))

        world_corners_list.append(world_corners)

        # Update bounds
        corner_xs, corner_ys = zip(*world_corners)
        all_west  = min(all_west, min(corner_xs))
        all_east  = max(all_east, max(corner_xs))
        all_south = min(all_south, min(corner_ys))
        all_north = max(all_north, max(corner_ys))

        # Close dataset
        ds = None

    # Add a small buffer (5%) to the bounds
    width      = all_east  - all_west
    height     = all_north - all_south
    all_west  -= width     * edge_width
    all_east  += width     * edge_width
    all_south -= height    * edge_width
    all_north += height    * edge_width

    # Create overlap grid if requested
    if show_overlap:
        # Create a grid covering the entire area
        aspect_ratio = width / height
        grid_width   = int(grid_size * aspect_ratio)
        grid_height  = grid_size
        grid_x       = np.linspace(all_west, all_east, grid_width)
        grid_y       = np.linspace(all_south, all_north, grid_height)
        X, Y         = np.meshgrid(grid_x, grid_y)

        # Create an empty overlap counter
        overlap_grid = np.zeros((grid_height, grid_width), dtype=int)

        # For each image, mark the grid cells it covers
        for corners in world_corners_list:
            path          = Path(corners)
            points        = np.column_stack([X.flatten(), Y.flatten()])
            mask          = path.contains_points(points).reshape(X.shape)
            overlap_grid += mask.astype(int)

    # Second pass: plot each image
    for i in range(first_idx, last_idx):
        # Get the file path from the rasterio dataset if needed
        if os.path.exists(image_list[i]["filepath"]):
            file_path = image_list[i]["filepath"]
        else:
            print(f"WARNING: File {image_list[i]['gdalImg']} does not exist")
            continue

        # Open with GDAL
        ds = gdal.Open(file_path)
        if not ds:
            print(f"WARNING: Could not open {file_path}")
            continue

        # Get geotransform
        gt = ds.GetGeoTransform()

        # Read the data
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()

        # Calculate corner coordinates for imshow
        width  = ds.RasterXSize
        height = ds.RasterYSize

        # Create a transformed image using pcolormesh for proper geotransform handling
        rows, cols = data.shape
        X = np.zeros((rows + 1, cols + 1))
        Y = np.zeros((rows + 1, cols + 1))

        # Create grid indices
        r_indices = np.arange(rows + 1)[:, np.newaxis]  # Column vector
        c_indices = np.arange(cols + 1)[np.newaxis, :]  # Row vector

        # Compute X and Y coordinates in one vectorized operation
        X = gt[0] + c_indices * gt[1] + r_indices * gt[2]
        Y = gt[3] + c_indices * gt[4] + r_indices * gt[5]

        # Handle masked data if needed
        if hasattr(data, 'mask'):
            data = np.ma.masked_where(data.mask, data)
        
        # Plot the image with proper transformation
        im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        
        # Close dataset
        ds = None
    
    # Plot overlap grid if requested
    if show_overlap and np.max(overlap_grid) > 1:
        # Create a custom colormap for overlap
        cmap_overlap = LinearSegmentedColormap.from_list(
            'overlap', 
            [(0.0, (0.0, 0.0, 0.0, 0.0)),  # Transparent for no overlap
             (0.5, (1.0, 0.0, 0.0, 0.3)),  # Semi-transparent red for single coverage
             (1.0, (1.0, 0.0, 0.0, 0.6))]  # More opaque red for multiple overlaps
        )
        
        # Normalize overlap grid to be between 0 and 1 for those with overlap
        norm_overlap = np.zeros_like(overlap_grid, dtype=float)
        mask         = overlap_grid > 1
        if np.any(mask):
            norm_overlap[mask] = (overlap_grid[mask] - 1) / (np.max(overlap_grid) - 1)

        # Plot the overlap grid
        ax.pcolormesh(grid_x, grid_y, norm_overlap, cmap=cmap_overlap, shading='auto')

        # Add legend for overlap
        import matplotlib.patches as mpatches
        legend_elements = [
            mpatches.Patch(facecolor=(1.0, 0.0, 0.0, 0.3), label='Overlap area')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    # Set extent
    ax.set_xlim(all_west, all_east)
    ax.set_ylim(all_south, all_north)

    # Add north arrow
    if show_north_arrow:
        _add_north_arrow(ax)

    # Calculate appropriate scale bar length
    if show_scale_bar:
        _add_scale_bar(ax, width, height, all_south, all_west)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')

    # Add axes labels
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

    if title:
        plt.title(title)

    # Add colorbar for main data
    if im is not None:
        plt.colorbar(im, ax=ax, orientation='vertical')
    plt.show()
    return fig, ax

def _add_north_arrow(ax, pos=(0.9, 0.1), size=0.1) -> None:
    """Add a north arrow to the plot"""
    arrow_x, arrow_y = pos
    arrow_length     = size
        
    # Draw arrow in axes coordinates
    ax.annotate('N', xy=(arrow_x, arrow_y), xycoords='axes fraction',
                xytext=(arrow_x, arrow_y - arrow_length), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=12, fontweight='bold')

    # Add scale bar
def _add_scale_bar(ax, width, height, all_south, all_west, location=(0.1, 0.05)) -> None:
    """
    Add a scale bar to the axes
    """
    nice_lengths = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    scale_length = width / 5  # Target: 1/5 of the image width
    length       = min(nice_lengths, key=lambda x: abs(x - scale_length))

    x0, y0 = all_west + location[0] * width, all_south + location[1] * height
    ax.plot([x0, x0 + length], [y0, y0], color='k', linewidth=3)
    ax.text(x0 + length/2, y0 + height/100, 
            f"{length:.0f} m", ha='center', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
def visualize_overlap(ds_1, ds_2) -> None:
    """
    Visualize the overlap between two georeferenced images.
    
    Parameters:
    img_name_n_1, img_name_n: Keys for the two images in self.images dictionary
    output_file: Optional path to save the plot. If None, plot is displayed.
    
    Returns:
    fig, ax: The matplotlib figure and axis objects
    """
    # Function to get corner coordinates in world space
   
    # Get corners for both images
    corners_1 = _get_corners(ds_1)
    corners_2 = _get_corners(ds_2)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the first image outline
    poly1  = Path(corners_1 + [corners_1[0]])  # Close the path
    patch1 = PathPatch(
        poly1, fill=False, edgecolor='blue', linewidth=2, 
        label = f"Image 1"
    )
    ax.add_patch(patch1)
    
    # Plot the second image outline
    poly2  = Path(corners_2 + [corners_2[0]])  # Close the path
    patch2 = PathPatch(
        poly2, fill=False, edgecolor='red', linewidth=2, 
        label=f"Image 2"
    )
    ax.add_patch(patch2)
    
    # Calculate the overlapping region using Shapely
    try:
        from shapely.geometry import Polygon
        
        # Convert corner lists to Shapely polygons
        poly1_shapely = Polygon(corners_1)
        poly2_shapely = Polygon(corners_2)
        
        # Calculate intersection
        intersection = poly1_shapely.intersection(poly2_shapely)
        
        if not intersection.is_empty:
            # Plot the overlap area
            if intersection.geom_type == 'Polygon':
                overlap_coords = list(intersection.exterior.coords)
                poly_overlap   = Path(overlap_coords)
                patch_overlap  = PathPatch(
                    poly_overlap, fill=True, facecolor='green', alpha=0.3, 
                    edgecolor='green', linewidth=2, label="Overlap"
                )
                ax.add_patch(patch_overlap)
                
                # Print overlap information
                overlap_area       = intersection.area
                total_area         = poly1_shapely.area + poly2_shapely.area - overlap_area
                overlap_percentage = (overlap_area / total_area) * 100
                
                # Find a good position for the text (use centroid of overlap)
                text_pos = intersection.centroid
                ax.text(
                    text_pos.x, text_pos.y,
                    f"Overlap: {overlap_area:.2f} sq units\n({overlap_percentage:.1f}% of total area)",
                    fontsize=10, color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
            else:
                # Handle multi-polygons if needed
                ax.text(
                    (min(corners_1[0][0], corners_2[0][0]) + max(corners_1[2][0], corners_2[2][0])) / 2,
                    max(corners_1[2][1], corners_2[2][1]),
                    "Complex overlap shape (multiple regions)",
                    fontsize=12, color='green', ha='center'
                )
        else:
            ax.text(
                (min(corners_1[0][0], corners_2[0][0]) + max(corners_1[2][0], corners_2[2][0])) / 2,
                max(corners_1[2][1], corners_2[2][1]),
                "No overlap between images",
                fontsize=12, color='red', ha='center'
            )
    except ImportError:
        ax.text(
            (min(corners_1[0][0], corners_2[0][0]) + max(corners_1[2][0], corners_2[2][0])) / 2,
            max(corners_1[2][1], corners_2[2][1]),
            "Install shapely to calculate overlap",
            fontsize=12, color='orange', ha='center'
        )
    
    # Set plot parameters
    ax.set_aspect('equal')
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    ax.set_title('Georeferenced Image Overlap Visualization (With Rotation)')
    ax.legend(loc='upper left')
    
    # Set axis limits to include both images
    all_corners = corners_1 + corners_2
    x_coords = [p[0] for p in all_corners]
    y_coords = [p[1] for p in all_corners]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add some padding
    padding = 0.1 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save or show plot
    
    plt.tight_layout()
    plt.show()

def _get_corners(ds) -> list:
    """
    Get the corner coordinates of a georeferenced image
    """
    height, width = ds.height, ds.width
    # Get pixel coordinates for all four corners
    corners_px = [(0, 0), (width, 0), (width, height), (0, height)]
    # Convert to world coordinates using the transform
    corners_world = [ds.transform * (x, y) for x, y in corners_px]
    return corners_world

def plot_offsets(offsets, confidence, mean_mag, std_div, title='Feature Matching Offsets', figsize=(10, 6)) -> None:
    """
    Plot feature matching offsets
    """
    offsets_x = [o[0] for o in offsets]
    offsets_y = [o[1] for o in offsets]
    # Create a side by side figure plot the x and y offsets data
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(title)
    # Plot x and y offsets
    axs[0].plot(offsets_x, label='X Offsets')
    axs[0].set_ylabel('X Offset (Pixels)')
    axs[0].set_xlabel('Image Index')
    axs[0].grid(True)
    axs[0].legend()
    # Plot y offsets
    axs[1].plot(offsets_y, label='Y Offsets')
    axs[1].set_ylabel('Y Offset (Pixels)')
    axs[1].set_xlabel('Image Index')
    axs[1].grid(True)
    axs[1].legend()

    ##Plot the confidence, mean magnitude and standard deviation
    fig, axs = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle(title)
    # Plot the confidence of the offsets
    axs[0].plot(confidence, label='Confidence')
    axs[0].set_ylabel('Confidence')
    axs[0].set_xlabel('Image Index')
    axs[0].grid(True)
    axs[0].legend()
    # Plot the mean magnitude of the offsets
    axs[1].plot(mean_mag, label='Mean Magnitude')
    axs[1].set_ylabel('Mean Magnitude')
    axs[1].set_xlabel('Image Index')
    axs[1].grid(True)
    axs[1].legend()
    # Plot the standard deviation of the offsets
    axs[2].plot(std_div, label='Standard Deviation')
    axs[2].set_ylabel('Standard Deviation')
    axs[2].set_xlabel('Image Index')
    axs[2].grid(True)
    axs[2].legend()
    # Show the plots
    plt.show()

def _visualize_transformations(offsets, rotations, skews, confidence) -> None:
    """
    Visualize all transformation parameters (translation, rotation, skew).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Prepare data
        image_indices = list(range(len(offsets)))
        x_shifts = [offset[0] for offset in offsets]
        y_shifts = [offset[1] for offset in offsets]
        rotations = rotations
        skew_x = [skew[0] for skew in skews]
        skew_y = [skew[1] for skew in skews]
        confidences = confidence
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # X and Y shifts
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(image_indices, x_shifts, c=confidences, cmap='viridis', 
                   alpha=0.8, s=50, edgecolors='k', linewidths=0.5)
        ax1.plot(image_indices, x_shifts, 'k-', alpha=0.3)
        ax1.set_title('X Shifts (meters)')
        ax1.set_xlabel('Image Index')
        ax1.set_ylabel('X Shift (m)')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(image_indices, y_shifts, c=confidences, cmap='viridis',
                   alpha=0.8, s=50, edgecolors='k', linewidths=0.5)
        ax2.plot(image_indices, y_shifts, 'k-', alpha=0.3)
        ax2.set_title('Y Shifts (meters)')
        ax2.set_xlabel('Image Index')
        ax2.set_ylabel('Y Shift (m)')
        ax2.grid(True, alpha=0.3)
        
        # Rotation
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(image_indices, rotations, c=confidences, cmap='viridis',
                   alpha=0.8, s=50, edgecolors='k', linewidths=0.5)
        ax3.plot(image_indices, rotations, 'k-', alpha=0.3)
        ax3.set_title('Rotation (degrees)')
        ax3.set_xlabel('Image Index')
        ax3.set_ylabel('Rotation (°)')
        ax3.grid(True, alpha=0.3)
        
        # Skew
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(image_indices, skew_x, c=confidences, cmap='viridis', marker='o',
                   alpha=0.8, s=50, edgecolors='k', linewidths=0.5, label='X Skew')
        ax4.scatter(image_indices, skew_y, c=confidences, cmap='plasma', marker='s',
                   alpha=0.8, s=50, edgecolors='k', linewidths=0.5, label='Y Skew')
        ax4.plot(image_indices, skew_x, 'k-', alpha=0.3)
        ax4.plot(image_indices, skew_y, 'k--', alpha=0.3)
        ax4.set_title('Skew (degrees)')
        ax4.set_xlabel('Image Index')
        ax4.set_ylabel('Skew (°)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Confidence
        ax5 = fig.add_subplot(gs[2, 0])
        bars = ax5.bar(image_indices, confidences, alpha=0.7, color='skyblue', edgecolor='navy')
        ax5.set_title('Transformation Confidence')
        ax5.set_xlabel('Image Index')
        ax5.set_ylabel('Confidence')
        ax5.set_ylim(0, 1.0)
        ax5.grid(True, alpha=0.3)
        
        # Combined vector field
        ax6 = fig.add_subplot(gs[2, 1])
        quiver_scale = 5.0
        quiver_width = 0.003
        
        # Scale shifts for visualization
        max_shift = max(max(abs(np.array(x_shifts))), max(abs(np.array(y_shifts))))
        if max_shift > 0:
            x_normalized = np.array(x_shifts) / max_shift
            y_normalized = np.array(y_shifts) / max_shift
        else:
            x_normalized = np.array(x_shifts)
            y_normalized = np.array(y_shifts)
        
        # Add vectors for shifts and rotations
        for i in range(len(image_indices)):
            # Base position
            x_pos = i
            y_pos = 0
            
            # Translation vector
            ax6.quiver(x_pos, y_pos, x_normalized[i], y_normalized[i], 
                      angles='xy', scale_units='xy', scale=quiver_scale, 
                      width=quiver_width, color='blue', alpha=0.7,
                      label='Translation' if i == 0 else "")
            
            # Rotation vector (using angle to determine direction)
            rotation_rad = np.radians(rotations[i])
            rot_x = 0.5 * np.sin(rotation_rad)
            rot_y = 0.5 * np.cos(rotation_rad)
            ax6.quiver(x_pos, y_pos, rot_x, rot_y, 
                      angles='xy', scale_units='xy', scale=quiver_scale,
                      width=quiver_width, color='red', alpha=0.7,
                      label='Rotation' if i == 0 else "")
            
            # Skew vectors
            skew_x_rad = np.radians(skew_x[i])
            skew_y_rad = np.radians(skew_y[i])
            
            # X skew vector
            skew_x_vec_x = 0.3 * np.cos(skew_x_rad)
            skew_x_vec_y = 0.3 * np.sin(skew_x_rad)
            ax6.quiver(x_pos, y_pos, skew_x_vec_x, skew_x_vec_y, 
                      angles='xy', scale_units='xy', scale=quiver_scale,
                      width=quiver_width, color='green', alpha=0.7,
                      label='X Skew' if i == 0 else "")
            
            # Y skew vector
            skew_y_vec_x = 0.3 * np.sin(skew_y_rad)
            skew_y_vec_y = 0.3 * np.cos(skew_y_rad)
            ax6.quiver(x_pos, y_pos, skew_y_vec_x, skew_y_vec_y, 
                      angles='xy', scale_units='xy', scale=quiver_scale,
                      width=quiver_width, color='purple', alpha=0.7,
                      label='Y Skew' if i == 0 else "")
        
        ax6.set_title('Combined Transformation Vectors')
        ax6.set_xlabel('Image Index')
        ax6.set_xlim(-0.5, len(image_indices) - 0.5)
        ax6.set_ylim(-1, 1)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_folder = self.config["MISSION"]["orthorectification_folder"]
        plt.savefig(os.path.join(output_folder, "transformation_parameters.png"), dpi=300)
        plt.close()
        
        print(f"Transformation visualization saved to {output_folder}/transformation_parameters.png")
        
    except Exception as e:
        print(f"Error visualizing transformations: {str(e)}")
        import traceback
        traceback.print_exc()

