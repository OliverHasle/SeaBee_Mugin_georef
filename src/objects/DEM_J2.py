import numpy             as np
import tools.constants   as c
import pyvista           as pv
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
class DEM_J2:
    def __init__(self, min_lat, max_lat, min_lon, max_lon, delta_ecef=100, resolution=100):
        """
        Create a DEM based on WGS-84 ellipsoid with J2 perturbation.

        Parameters:
        -----------
        max_lat, min_lat : float
            Latitude boundaries in degrees
        max_lon, min_lon : float
            Longitude boundaries in degrees
        resolution : int
            Number of points along each axis
        """
        self.delta_lla   = None
        self.j2_spheroid = None
        self.j2_section  = None
        # WGS84 parameters
        self.lat_lon_box = [min_lat, max_lat, min_lon, max_lon]
        self.delta_ecef  = delta_ecef                  # Distance in ECEF coordinates
        self.resolution  = resolution

        self._delta_lla()                              # Distance in lat/lon coordinates
        self._create_j2_spheroid()                     # Create J2 spheroid mesh
        #self.j2_spheroid.plot()
        self._create_j2_section()                      # Create section of the J2 spheroid mesh

    def _delta_lla(self):
        """
        Convert a certain distance in ECEF coordinates into the equivalent distance in lat/lon coordinates (delta lat/lon).
        """
        # Calculate "delta_lat" from the distance delta_ecef
        delta_lat = self.delta_ecef * 180 / (np.pi * c.EARTH_RADIUS)

        # Delta longitude (using actual reference latitude lat0)
        ref_lat   = self.lat_lon_box[0] + (self.lat_lon_box[1] - self.lat_lon_box[0]) / 2
        delta_lon = np.rad2deg(self.delta_ecef / (c.EARTH_RADIUS * np.cos(np.radians(ref_lat))))    
        delta_alt = self.delta_ecef  # Assuming a simple altitude offset
    
        self.delta_lla = np.array([delta_lat, delta_lon, delta_alt])

    def _create_j2_spheroid(self, theta_res=10000, phi_res=10000):
        """
        Create a PyVista mesh representing Earth's J2 spheroid
        
        Args:
            R_eq: Equatorial radius (default WGS84)
            f: Flattening factor (default WGS84)
        """
        # Create sphere
        sphere = pv.Sphere(radius=c.EARTH_RADIUS, theta_resolution=theta_res, phi_resolution=phi_res)

        # Modify vertices to create J2 spheroid shape
        vertices = sphere.points.copy()

        # Apply flattening at poles
        vertices[:, 2] *= (1 - c.FLATTENING)

        # Recreate mesh with modified vertices
        self.j2_spheroid = pv.PolyData(vertices, sphere.faces)

    def _create_j2_section(self):
        """
        Extract a subset of the spheroid based on latitude and longitude limits

        Args:
        spheroid: PyVista mesh
        lat_min, lat_max: Latitude range in degrees
        lon_min, lon_max: Longitude range in degrees
        """
        # Convert degrees to radians
        lat_min, lat_max = np.deg2rad(self.lat_lon_box[0:2])    #np.radians([lat_min, lat_max])
        lon_min, lon_max = np.deg2rad(self.lat_lon_box[2:4])    #np.radians([lon_min, lon_max])

        # Calculate spherical coordinates
        points = self.j2_spheroid.points
        lats    = np.arcsin(points[:, 2] / np.linalg.norm(points, axis=1))
        lons    = np.arctan2(points[:, 1], points[:, 0])

        # Create mask for points within specified range
        mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)

        # Extract subset
        self.mesh = self.j2_spheroid.extract_points(mask)
