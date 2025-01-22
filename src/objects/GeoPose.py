import os
import pyproj

import numpy   as np

class GeoPose:
    def __init__(self, config, parameter):
        self.gpslogpath = None
        self.latitude   = []
        self.longitude  = []
        self.altitude   = []
        self.roll       = []
        self.pitch      = []
        self.yaw        = []
        self.p_ec_e     = None
        self.R_e_b      = None

        self.b_c_b      = None
        self.fov_x      = None
        self.fov_y      = None

        self._initialize(config, parameter)

    def _initialize(self, config, parameter):
        # Open gpslog file
        gpslogfile = [filename for filename in os.listdir(config['MISSION']['inputfolder']) if filename.startswith("gpslog")]
        self.gpslogpath = os.path.join(config['MISSION']['inputfolder'], gpslogfile[0])  # Use the first gpslog file

        # Initialize an empty list to store the GPS data
        gpslog = []

        # Open and read the GPS log line by line
        with open(self.gpslogpath, 'r') as file:
            for line in file:
                values    = line.strip().split(',')
                gps_entry = {
                    "filename":  values[0],
                    "latitude":  float(values[1]),
                    "longitude": float(values[2]),
                    "altitude":  float(values[3]),
                    "roll":      float(values[4]),
                    "pitch":     float(values[5]),
                    "yaw":       float(values[6])
                }
                gpslog.append(gps_entry)

        # Example usage for each image entry
        for entry in gpslog:
#            filename  = entry["filename"]
            self.latitude.append(entry["latitude"])
            self.longitude.append(entry["longitude"])
            self.altitude.append(entry["altitude"])
            self.roll.append(entry["roll"])
            self.pitch.append(entry["pitch"])
            self.yaw.append(entry["yaw"])

        self.b_c_b = np.array([0, 0, 1])    # Calculate the boresight vector b_c_e in ECEF

        self.fov_x = self.calc_fov(parameter['sensor']['focal_length'], parameter['sensor']['sensor_dimensions']['width'])
        self.fov_y = self.calc_fov(parameter['sensor']['focal_length'], parameter['sensor']['sensor_dimensions']['height'])

    def calculate_camera_position(self, lat, lon, alt, roll, pitch, yaw, param):
        # Convert the GPS coordinates to WGS84, ECEF
        p_eb_e = self._lla_to_ecef(lat, lon, alt)
        p_bc_b = param['p_bc_b']

        # Calculate the rotation matrix body -> ECEF from roll, pitch, yaw
        R_e_b = self._calc_R_eb(roll, pitch, yaw)

        p_ec_e = p_eb_e + R_e_b @ p_bc_b
        return p_ec_e, R_e_b
    
    def calculate_intersection(self, dem, config, p_ec_e, R_e_b, b_c_b):
        pass

    @staticmethod
    def _lla_to_ecef(lat, lon, alt):
        # Convert the GPS coordinates to WGS84, ECEF
        ecef    = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla     = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
        return np.array([x, y, z])

    @staticmethod
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
    
    """
    Optical calculations
    """
    @staticmethod
    def calc_fov(focal_length, sensor_dimension):
        # Convert focal length and sensor dimensions from mm to meters
        focal_length_m     = np.float64(focal_length)     / 1000.0
        sensor_dimension_m = np.float64(sensor_dimension) / 1000.0
        # Calculate field of view in radians
        return 2 * np.arctan(sensor_dimension_m / (2 * focal_length_m))
