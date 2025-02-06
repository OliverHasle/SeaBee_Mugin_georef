import numpy           as np
import tools.constants as c
import pyvista         as pv
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
        self.delta_lla = None
        # WGS84 parameters
        self.lat_lon_box = [min_lat, max_lat, min_lon, max_lon]
        self.delta_ecef  = delta_ecef                  # Distance in ECEF coordinates
        self.resolution = resolution
        self._delta_lla()  # Distance in lat/lon coordinates

        # Create grid in lat/lon
        lats = np.linspace(self.lat_lon_box[0] - self.delta_lla[0], self.lat_lon_box[1] + self.delta_lla[0], self.resolution)
        lons = np.linspace(self.lat_lon_box[2] - self.delta_lla[1], self.lat_lon_box[3] + self.delta_lla[1], self.resolution)
        # Create meshgrid
        self.LON, self.LAT = np.meshgrid(np.deg2rad(lons), np.deg2rad(lats))
#self._calc_mesh()
        # Calculate radius with J2 perturbation
        sin_lat    = np.sin(self.LAT)
        sin_lat_sq = sin_lat * sin_lat

        # Base ellipsoid radius
        r_ellipsoid = c.EARTH_RADIUS * (1 - c.EXCENTRICITY_2) / np.sqrt(1 - c.EXCENTRICITY_2 * sin_lat_sq)

        # J2 perturbation
        P2      = 0.5 * (3 * sin_lat_sq - 1)  # Second Legendre polynomial
        delta_r = -c.EARTH_RADIUS * c.EARTH_J2 * P2

        # Total radius including J2
        r = r_ellipsoid + delta_r

        # Convert to ECEF coordinates
        cos_lat = np.cos(self.LAT)
        cos_lon = np.cos(self.LON)
        sin_lon = np.sin(self.LON)

        X = r * cos_lat * cos_lon
        Y = r * cos_lat * sin_lon
        Z = r * sin_lat

        # Create points array
        self.points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

        # Create triangulation in lat/lon space (easier to triangulate on 2D surface)
        latlon_points = np.column_stack((self.LON.flatten(), self.LAT.flatten()))
        self.triangulation = Delaunay(latlon_points)

        # Create mesh attribute
        self.simp_mesh = self.SimpleMesh(self.points, self.triangulation)

        # Make the "SimpleMesh" an pyvista object
        self.mesh = pv.PolyData(self.points, self.triangulation.simplices)

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

        self.delta_lla = np.array([delta_lat, delta_lon, delta_alt])

    class SimpleMesh:
        def __init__(self, points, triangulation):
            self.points        = points
            self.triangulation = triangulation
        
        def multi_ray_trace(self, origins, directions, first_point=True):
            """
            Ray tracing for ellipsoid surface.
            
            Parameters:
            -----------
            origins : ndarray
                Array of ray origin points in ECEF coordinates
            directions : ndarray
                Array of ray directions in ECEF coordinates
            first_point : bool
                Return only first intersection point
            
            Returns:
            --------
            points : ndarray
                Intersection points
            rays : ndarray
                Ray indices
            cells : ndarray
                Cell indices
            """
            num_rays = origins.shape[0]
            points = np.zeros_like(origins)
            rays = np.arange(num_rays)
            cells = np.zeros(num_rays, dtype=int)
            
            # For each triangle in the mesh
            for i, simplex in enumerate(self.triangulation.simplices):
                vertices = self.points[simplex]
                
                # Calculate triangle normal
                v1 = vertices[1] - vertices[0]
                v2 = vertices[2] - vertices[0]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                
                for j in range(num_rays):
                    # Ray-plane intersection
                    d = np.dot(normal, vertices[0] - origins[j])
                    denom = np.dot(normal, directions[j])
                    
                    if abs(denom) > 1e-10:  # Check if ray is not parallel to plane
                        t = d / denom
                        if t > 0:  # Check if intersection is in front of origin
                            intersection = origins[j] + t * directions[j]
                            
                            # Check if point is inside triangle
                            if self.point_in_triangle(intersection, vertices):
                                points[j] = intersection
                                cells[j] = i
            
            # Filter out rays that didn't intersect
            valid = cells >= 0
            return points[valid], rays[valid], cells[valid]
        
        @staticmethod
        def point_in_triangle(point, triangle_vertices):
            """Check if point is inside triangle using barycentric coordinates."""
            v0 = triangle_vertices[1] - triangle_vertices[0]
            v1 = triangle_vertices[2] - triangle_vertices[0]
            v2 = point - triangle_vertices[0]
            
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)
            
            invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom
            
            return (u >= 0) and (v >= 0) and (u + v <= 1)
