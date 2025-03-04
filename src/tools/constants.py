"""
This file contains constatnts used in the project.
"""

EARTH_RADIUS              = 6378137.0                                    # Earth radius of the WGS84 ellipsoid in meters / Semi-major axis
EARTH_SEMI_MAJOR_AXIS     = EARTH_RADIUS                                 # Earth semi-major axis of the WGS84 ellipsoid in meters
FLATTENING                = 1/298.257223563                              # flattening
SEMI_MINOR_AXIS           = EARTH_SEMI_MAJOR_AXIS * (1 - FLATTENING)     # semi-minor axis
EXCENTRICITY_2            = 2*FLATTENING - FLATTENING**2                 # eccentricity 2*self.f - self.f**2
EARTH_J2                  = 1.08263e-3                                   # J2 perturbation coefficient for Earth 1.08263e-3

EPSG_latlon  = 4326                                                      # EPSG code for WGS84 lat/lon coordinate system
EPSG_geocent = 4978                                                      # EPSG code for WGS84 geocentric coordinate system