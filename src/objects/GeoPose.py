import os
import time
import pyproj

import numpy   as np

class GeoPose:
    def __init__(self, config, parameter):
        self.config     = config
        self.gpslogpath = None
        self.no_images  = None
        self.no_meas    = None
        self.latitude   = None
        self.longitude  = None
        self.altitude   = None
        self.roll       = None
        self.pitch      = None
        self.yaw        = None

        self.p_eb_e     = None
        self.p_ec_e     = None
        self.R_b_c      = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) # Rotation matrix from camera to body
        self.R_e_b      = None

        self.p_bc_b     = None
        self.l_cd_max   = None

        self.v_c_b      = np.array([0, 0, 1])                          # Boresight vector b_c_e in ECEF
        self.fov_x      = None
        self.fov_y      = None

        self.v_c_c      = None  # Ray direction of the camera in camera coordinates [0 = camera center, 1= upper left corner, 2 = upper right corner, 3 = lower left corner, 4 = lower right corner]
        self.v_c_e      = None  # Ray direction of the camera in ECEF [0 = camera center, 1= upper left corner, 2 = upper right corner, 3 = lower left corner, 4 = lower right corner]
        self.p_eg_e     = []    # Ground point in ECEF

        self._initialize(config, parameter)

    def _initialize(self, config, parameter):
        # Open gpslog file
        gpslogfile      = [filename for filename in os.listdir(config['MISSION']['inputfolder']) if filename.startswith("gpslog")]
        self.gpslogpath = os.path.join(config['MISSION']['inputfolder'], gpslogfile[0])  # Use the first gpslog file

        # Initialize an empty list to store the GPS data
        self._read_gpslog(config)

        self.fov_x = self._calc_fov(parameter['sensor']['focal_length'], parameter['sensor']['sensor_dimensions']['width'])
        self.fov_y = self._calc_fov(parameter['sensor']['focal_length'], parameter['sensor']['sensor_dimensions']['height'])

        # Get the camera parameters
        self.p_bc_b   = parameter['p_bc_b']
        self.l_cd_max = parameter['l_cd']


    def calculate_camera_position(self):
        """
            TBD
        """
        # Convert the GPS coordinates to WGS84, ECEF
        self.p_eb_e = self._lla_to_ecef(self.latitude, self.longitude, self.altitude)
        
        # Calculate the rotation matrix body -> ECEF from roll, pitch, yaw
        self.R_e_b = self._calc_R_eb(self.roll, self.pitch, self.yaw, self.no_meas)

        # Calculate the camera position in ECEF
        self.p_ec_e = np.zeros((self.no_meas, 3))
        for i in range(self.no_meas):
            self.p_ec_e[i,:] = self.p_eb_e[i,:] + self.R_e_b[i,:,:] @ self.p_bc_b

  
    def boresight_mesh_intersection(self, dem):
        """
            TBD
        """

        print('Georeferencing Images')

        # Calculate Ray Directions in ECEF

        self._camera_properties()
        if self.v_c_e is None or self.v_c_c is None:
            raise ValueError("Camera properties have not been calculated")
        self.p_eg_e             = np.zeros((self.no_meas, len(self.v_c_c), 3), dtype=np.float64)
        self.normals_ecef_crs   = np.zeros((self.no_meas, len(self.v_c_c), 3), dtype=np.float64) # ???

        start       = np.einsum('ijk, ik -> ijk', np.ones((self.no_meas, len(self.v_c_c), 3), dtype=np.float64), self.p_ec_e).reshape((-1,3))      # 2D array of the camera postions (540 x the same "starting position" for each ray) => (2000 * 540) x 3 -> 2D array
        camera_rays = (self.v_c_e * self.l_cd_max).reshape((-1,3))                                                                                      # All global ray directions (2000 * 540) x 3 -> 2D array
        # points = The intersection points of the rays with the mesh
        # rays   = The ray indices (numberes from 0 to no_of_rays)
        # cells  = the cell numbers of the intersected cells (several rays can intersect the same cell)
        start_time          = time.time()
        self.p_eg_e, rays, cells = dem.mesh.multi_ray_trace(origins=start, directions=camera_rays, first_point=True)
        stop_time           = time.time()

        print(f"Ray tracing finished")
        print(f"Time for ray tracing: {stop_time - start_time} seconds")

    """
    PRIVATE METHODS
    """
    def _read_gpslog(self, config):
        gpslog = np.genfromtxt(self.gpslogpath, delimiter=',',
                                dtype=[('filename',  'U100'),
                                       ('latitude',  'f8'),
                                       ('longitude', 'f8'),
                                       ('altitude',  'f8'),
                                       ('roll',      'f8'),
                                       ('pitch',     'f8'),
                                       ('yaw',       'f8'
                                       )])

        self.latitude  = gpslog['latitude']
        self.longitude = gpslog['longitude']
        self.altitude  = gpslog['altitude']
        self.roll      = gpslog['roll']
        self.pitch     = gpslog['pitch']
        self.yaw       = gpslog['yaw']

        # ROBUSTNESS CHECK: Check if the number of images is equal to the number of GPS entries
        self.no_images = len([f for f in os.listdir(config['MISSION']['inputfolder']) if f.endswith('.JPG')])
        if len(self.latitude) != len(self.longitude) or len(self.latitude) != len(self.altitude) or len(self.latitude) != len(self.roll) or len(self.latitude) != len(self.pitch) or len(self.latitude) != len(self.yaw):
            raise ValueError("Values in the GPS log do have different dimensions")
        else:
            self.no_meas = len(self.latitude)

        if self.no_images != self.no_meas:
            print(f"WARNING: The number of images: {self.no_images} does not match the number of GPS {self.no_meas} entries")

    def _camera_properties(self):
        """
        Return an array of vectors pointing to the corners of the image sensor
        """
        # Calculate the half field of view
        half_fov_x = self.fov_x / 2
        half_fov_y = self.fov_y / 2

        # Calculate the corner vectors of the image sensor (in camera coordinates)
        v_c_c = np.array([
            [np.tan(half_fov_x), -np.tan(half_fov_y), 1],
            [np.tan(half_fov_x), np.tan(half_fov_y), 1],
            [-np.tan(half_fov_x), -np.tan(half_fov_y), 1],
            [+np.tan(half_fov_x), np.tan(half_fov_y), 1],
            [0, 0, 1]
        ])

        # Normalize the vectors
        for i in range(len(v_c_c)):
            v_c_c[i] = v_c_c[i] / np.linalg.norm(v_c_c[i])

        # Calculate the pointing of the vectors v_c_c in ECEF, and normalize them to unit vectors
        # v_c_e = R_e_b @ v_c_c
        v_c_e = np.zeros((self.no_meas, len(v_c_c), 3))
        for i in range(self.no_meas):
            for j in range(len(v_c_c)):
                v_c_e[i,j,:] = self.R_e_b[i,:,:] @ self.R_b_c @ v_c_c[j,:]
                v_c_e[i,j,:] = v_c_e[i,j,:] / np.linalg.norm(v_c_e[i,j,:])
#                v_c_e[i,j,:] = self.R_e_b[i,:,:] @ self.R_b_c @ v_c_c[j,:] / np.linalg.norm(self.R_e_b[i,:,:] @ v_c_c[j,:])

        self.v_c_c = v_c_c
        self.v_c_e = v_c_e
        

    @staticmethod
    def _lla_to_ecef(lat, lon, alt):
        # Convert the GPS coordinates to WGS84, ECEF
        ecef    = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla     = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
        return np.array([x, y, z]).reshape(-1, 3)

    @staticmethod
    def _calc_R_eb(roll, pitch, yaw, no_meas):
        # Calculate the rotation matrix body -> ECEF from roll, pitch, yaw
        # Check if roll pitch and yaw have the same length
        R_e_b = np.zeros((no_meas, 3, 3))

        # Calculate the roation matrix for each time step
        for i in range(no_meas):
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(roll[i]), -np.sin(roll[i])],
                [0, np.sin(roll[i]), np.cos(roll[i])]
            ])
            R_y = np.array([
                [np.cos(pitch[i]), 0, np.sin(pitch[i])],
                [0, 1, 0],
                [-np.sin(pitch[i]), 0, np.cos(pitch[i])]
            ])
            R_z = np.array([
                [np.cos(yaw[i]), -np.sin(yaw[i]), 0],
                [np.sin(yaw[i]), np.cos(yaw[i]), 0],
                [0, 0, 1]
            ])
            R_e_b[i,:,:] = R_z @ R_y @ R_x
        return R_e_b
             
#        R_x = np.array([
#            [np.ones_like(roll), np.zeros_like(roll), np.zeros_like(roll)],
#            [0, np.cos(roll), -np.sin(roll)],
#            [0, np.sin(roll), np.cos(roll)]
#        ])
#        R_y = np.array([
#            [np.cos(pitch), 0, np.sin(pitch)],
#            [0, 1, 0],
#            [-np.sin(pitch), 0, np.cos(pitch)]
#        ])
#        R_z = np.array([
#            [np.cos(yaw), -np.sin(yaw), 0],
#            [np.sin(yaw), np.cos(yaw), 0],
#            [0, 0, 1]
#        ])
#        return R_z @ R_y @ R_x

    """
    Optical calculations
    """
    @staticmethod
    def _calc_fov(focal_length, sensor_dimension):
        # Convert focal length and sensor dimensions from mm to meters
        focal_length_m     = np.float64(focal_length)     / 1000.0
        sensor_dimension_m = np.float64(sensor_dimension) / 1000.0
        # Calculate field of view in radians
        return 2 * np.arctan(sensor_dimension_m / (2 * focal_length_m))
    
    
