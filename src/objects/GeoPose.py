import os
import time
import pyproj
import numpy                       as np
import tools.coordinateConversions as cc

debug = False

class GeoPose:
    def __init__(self, config, parameter)      -> None:
        self.config        = config # Configuration file
        self.gpslogpath    = None   # Path to the GPS log file
        self.image_name    = None   # Image names in the GPS log file
        self.no_images     = None   # Number of images in the input folder (JPG files)
        self.no_meas       = None   # Number of GPS measurements in the GPS log file
        self.latitude      = None   # Latitude in degrees
        self.longitude     = None   # Longitude in degrees
        self.baro_alt      = None   # Barometeric altitude in meters
        self.altitude      = None   # Altitude in meters above WGS84 ellipsoid
        self.roll          = None   # Roll in degrees
        self.pitch         = None   # Pitch in degrees       
        self.yaw           = None   # Yaw in degrees

        self.p_eb_e        = None                                         # Position of the body in ECEF               [no_meas x 3]
        self.p_ec_e        = None                                         # Position of the camera in ECEF             [no_meas x 3]

#        self.R_b_c         = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) # Rotation matrix from camera to body frame  [no_meas x 3 x 3]
        self.R_n_b         = None                                         # Rotation matrix from body to NED           [no_meas x 3 x 3]
        self.q_n_b         = None                                         # Quaternion from body to NED                [no_meas x 4]
        self.R_e_n         = None                                         # Rotation matrix from NED to ECEF           [no_meas x 3 x 3]

        self.camera_angles = parameter['camera_angles']                   # Camera angles in degrees                   [no_meas x 3]        
        self.p_bc_b        = parameter['p_bc_b']                          # Lever arm from body to camera in body coordinates [3]
        self.l_cd_max      = parameter['l_cd']                            # Maximum distance from camera to the ground in meters, distanced grater than this will be ignored [1]

        self.fov_x         = None                                         # Field of view in x direction in radians  [1] (along track)
        self.fov_y         = None                                         # Field of view in y direction in radians  [1] (cross track)

        self.v_c_b               = None                                   # Ray direction of the camera in body coordinates [0=upper left corner, 1=upper right corner, 2=lower left corner, 3=lower right corner, 4=camera center] [5 x 3]
        self.v_c_e               = None                                   # Ray direction of the camera in ECEF             [0=upper left corner, 1=upper right corner, 2=lower left corner, 3=lower right corner, 4=camera center] [no_meas x 5 x 3]
        self.p_eg_e              = None                                   # Ground point in ECEF                            [0=upper left corner, 1=upper right corner, 2=lower left corner, 3=lower right corner, 4=camera center] [no_meas x 5 x 3]
        self.p_eg_e_cell_normals = None                                   # Normal vectors of the intersected cells in ECEF                                                                                                         [no_meas x 5 x 3]   
        self.intersect_image_no  = None                                   # Image number of the intersected ray                                                                                                                     [no_meas x 5] 

        self._initialize(config, parameter)

    def _initialize(self, config, parameter)   -> None:
        # Open gpslog file
        gpslogfile      = [filename for filename in os.listdir(config['MISSION']['inputfolder']) if filename.startswith("gpslog")]
        self.gpslogpath = os.path.join(config['MISSION']['inputfolder'], gpslogfile[0])  # Use the first gpslog file

        # Initialize an empty list to store the GPS data
        self._read_gpslog(config)
        self._data_preprocessing(config)

        self.fov_x = self._calc_fov(parameter['sensor']['focal_length'], parameter['sensor']['sensor_dimensions']['height'])     # Field of view in x direction in radians
        self.fov_y = self._calc_fov(parameter['sensor']['focal_length'], parameter['sensor']['sensor_dimensions']['width'])    # Field of view in y direction in radians

        # Convert the GPS coordinates to WGS84, ECEF
        self._lla_to_ecef()
        self._calc_R_en()
        self._calc_q_en()

        # Calculate the rotation matrix body -> ECEF from roll, pitch, yaw
        self._euler2R_nb()
        self._euler2q_nb()
#        self._R_nb2q_nb()

    def calculate_camera_position(self)        -> None:
        """
        This function calculates the camera position in ECEF coordinates taking into account the 
        UAV position, the camera mounting angle, the camera lever arm on the UAV and the UAV attitude.
        """
        # Calculate the camera position in ECEF
        self.p_ec_e = np.zeros((self.no_meas, 3))
        for i in range(self.no_meas):
            self.p_ec_e[i,:] = self.p_eb_e[i,:] + self.R_e_n[i,:,:] @ self.R_n_b[i,:,:] @ self.p_bc_b
        # Protect numpy array agains alteration (read-only)
        self.p_ec_e.flags.writeable = False

    def boresight_mesh_intersection(self, dem) -> None:
        """
        Calculate the intersection of the camera rays with the ground (DEM) mesh
        This funciton updates the following class variables:
            - p_eg_e:              Ground point in ECEF
            - p_eg_e_cell_normals: Normal vectors of the intersected cells in ECEF
            - intersect_mask:      Mask indicating if the ray intersects the DEM/mesh
            - intersect_image_no:  Image number of the intersected ray
        """
        print('Georeferencing Images')

        # Calculate Ray Directions in ECEF
        self._camera_properties()
        if self.v_c_e is None or self.v_c_b is None:
            raise ValueError("Camera properties have not been calculated")
        
        num_rays = len(self.v_c_b)

        ray_start_pos           = np.einsum('ijk, ik -> ijk',
                                             np.ones((self.no_meas, len(self.v_c_b), 3), dtype=np.float64), 
                                             self.p_ec_e).reshape((-1,3))                                              # 2D array of the camera postions (5 x the same "starting position" for each ray) => (no_meas * 5) x 3 -> 2D array
        camera_rays_dir         = (self.v_c_e * self.l_cd_max).reshape((-1,3))                                         # All global ray directions (no_meas * 5) x 3 -> 2D array

        # intersects = The intersection points of the rays with the mesh
        # rays       = The ray indices (numberes from 0 to no_of_rays)
        # cells      = the cell numbers of the intersected cells (several rays can intersect the same cell)
        start_time               = time.time()
        intersects, rays, cells  = dem.mesh.multi_ray_trace(origins=ray_start_pos, directions=camera_rays_dir, first_point=True)
        stop_time                = time.time()

        # Initialize the intersection points, the cell normals and the intersect mask with NaN/False
        self.p_eg_e              = np.full((self.no_meas, len(self.v_c_b), 3), np.nan, dtype=np.float64)
        self.p_eg_e_cell_normals = np.full((self.no_meas, len(self.v_c_b), 3), np.nan, dtype=np.float64)
        self.intersect_mask      = np.zeros((self.no_meas, len(self.v_c_b)), dtype=bool)

        if len(intersects) > 0:
            measurement_indices = rays // num_rays
            camera_ray_indices  = rays % num_rays

            # Use advanced indexing for fast assignment
            self.p_eg_e[measurement_indices, camera_ray_indices]               = intersects
            self.p_eg_e_cell_normals[measurement_indices, camera_ray_indices]  = dem.mesh.cell_normals[cells]
            self.intersect_mask[measurement_indices, camera_ray_indices]       = True
            self.intersect_image_no                                            = measurement_indices

            # Make arrays read-only
            for arr in [self.p_eg_e, self.p_eg_e_cell_normals, 
                        self.intersect_mask, self.intersect_image_no]:
                arr.flags.writeable = False
        else:
            print(f"----------------------------")
            print(f"No intersection points found")
            print(f"----------------------------")

        print(f"Ray tracing finished")
        print(f"Time for ray tracing: {stop_time - start_time} seconds")

        if debug == True:
            index_of_interest = 200
            alt_abv_geoid     = np.linalg.norm(self.p_eg_e[index_of_interest,4,:] - self.p_ec_e[index_of_interest,:])
            print(f"altitude above geoid: {alt_abv_geoid}")
            print(f"Barometer altitude: {self.baro_alt[index_of_interest]}")

    """
    PRIVATE METHODS
    """
    def _read_gpslog(self, config)             -> None:
        gpslog = np.genfromtxt(self.gpslogpath, delimiter=',',
                                dtype=[('filename',  'U100'),
                                       ('latitude',  'f8'),
                                       ('longitude', 'f8'),
                                       ('altitude',  'f8'),
                                       ('roll',      'f8'),
                                       ('pitch',     'f8'),
                                       ('yaw',       'f8'
                                       )])

        self.image_name = gpslog['filename']
        self.latitude   = gpslog['latitude']
        self.longitude  = gpslog['longitude']
        self.baro_alt   = gpslog['altitude']
        self.roll       = gpslog['roll']
        self.pitch      = gpslog['pitch']
        self.yaw        = gpslog['yaw']

        # Protect numpy array agains alteration (read-only)
        self.image_name.flags.writeable = False
        self.latitude.flags.writeable   = False
        self.longitude.flags.writeable  = False
        self.baro_alt.flags.writeable   = False
        self.roll.flags.writeable       = False
        self.pitch.flags.writeable      = False
        self.yaw.flags.writeable        = False

        # ROBUSTNESS CHECK: Check if the number of images is equal to the number of GPS entries
        self.no_images = len([f for f in os.listdir(config['MISSION']['inputfolder']) if f.lower().endswith('.png') or f.lower().endswith('.jpg')])
        if len(self.latitude) != len(self.longitude) or len(self.latitude) != len(self.baro_alt) or len(self.latitude) != len(self.roll) or len(self.latitude) != len(self.pitch) or len(self.latitude) != len(self.yaw):
            raise ValueError("Values in the GPS log do have different dimensions")
        else:
            self.no_meas = len(self.latitude)

        if self.no_images != self.no_meas:
            print(f"--------------------------------------------------------------------------------------------------------------")
            print(f"WARNING: The number of images: {self.no_images} does not match the number of GNSS measurements: {self.no_meas}")
            print(f"--------------------------------------------------------------------------------------------------------------")

    def _data_preprocessing(self, config)      -> None:
        self.altitude = self.baro_alt + np.float32(config['SETTINGS']['wgs84_altitude_at_takeoff'])  + np.float32(config['TUNING']['delta_altitude']) # Altitude in meters above WGS84 ellipsoid
    def _camera_properties(self)               -> None:
        """
        Return an array of vectors pointing to the corners of the image sensor
        """
        # Calculate the half field of view
        half_fov_x = self.fov_x / 2
        half_fov_y = self.fov_y / 2

        # Calculate the corner vectors of the image sensor (in camera coordinates)
        v_c_b_nadir = np.array([
            [np.tan(half_fov_x), -np.tan(half_fov_y), 1],
            [np.tan(half_fov_x), np.tan(half_fov_y), 1],
            [-np.tan(half_fov_x), -np.tan(half_fov_y), 1],
            [-np.tan(half_fov_x), np.tan(half_fov_y), 1],
            [0, 0, 1]
        ])

        # Tuning values
        pitch = np.float32(self.config['CAMERA POSITION ADJUSTMENT']['pitch_angle'])
        roll  = np.float32(self.config['CAMERA POSITION ADJUSTMENT']['roll_angle'])
        yaw   = np.float32(self.config['CAMERA POSITION ADJUSTMENT']['yaw_angle'])

        # Adjust the vectors for the camera mounting angle and the UAV trimmed flight conditions
        R_tuning = cc.euler2Rot(roll, pitch, yaw, unit='deg')
        v_c_b    = np.zeros((len(v_c_b_nadir), 3))
        for i in range(len(v_c_b_nadir)):
            v_c_b[i] = R_tuning @ v_c_b_nadir[i,:]
            # Normalize the vectors
            v_c_b[i] = v_c_b[i] / np.linalg.norm(v_c_b[i])

        # Calculate the pointing of the vectors v_c_b in ECEF, and normalize them to unit vectors
        # v_c_e = R_n_b @ v_c_b
        v_c_e = np.zeros((self.no_meas, len(v_c_b), 3))

        # Loop over all measurements
        for i in range(self.no_meas):
            # Loop over all vectors originating from the camera
            for j in range(len(v_c_b)):
                v_c_e[i,j,:] = self.R_e_n[i,:,:] @ self.R_n_b[i,:,:] @ v_c_b[j,:]
                #v_c_e[i,j,:] = self.R_e_n[i,:,:] @ v_c_b[j,:]                           # DEBUG: for debugging (assumption that R_n_b is the identity matrix, UAV is always pointing north and the camera is always pointing down)
                v_c_e[i,j,:] = v_c_e[i,j,:] / np.linalg.norm(v_c_e[i,j,:])

        self.v_c_b = v_c_b
        self.v_c_e = v_c_e
        self.v_c_b.flags.writeable = False
        self.v_c_e.flags.writeable = False        

    def _lla_to_ecef(self)                     -> None:
        # Convert the GPS coordinates to WGS84, ECEF
        ecef    = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84') # EPSG-4326
        lla     = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84') # EPSG-4978
        x, y, z = pyproj.transform(lla, ecef, self.longitude, self.latitude, self.altitude, radians=False)
    
        p_eb_e = np.zeros((self.no_meas, 3))
        p_eb_e[:,0] = x
        p_eb_e[:,1] = y
        p_eb_e[:,2] = z
        self.p_eb_e = p_eb_e
        self.p_eb_e.flags.writeable = False
   
    """
    STATIC METHODS
    """
    @staticmethod
    def lat_lon_h2p_eb_e(lat_deg, lon_deg, h)  -> np.ndarray:
        # Convert latitude and longitude to radians
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)

        # WGS84 parameters
        a = 6378137  # semi-major axis
        f = 1/298.257223563  # flattening of the Earth-ellipsoid
    
        # Compute eccentricity squared
        e_sq = 2*f - f**2
    
        # Compute normal/prime vertical radius
        R_N = a / np.sqrt(1 - e_sq * np.sin(lat)**2)
    
        # Compute ECEF coordinates
        p_eb_e = np.array([
            (R_N + h) * np.cos(lat) * np.cos(lon),
            (R_N + h) * np.cos(lat) * np.sin(lon),
            (R_N * (1 - e_sq) + h) * np.sin(lat)
        ])
        return p_eb_e
    def _calc_R_en(self)                       -> None:
        # Calculate the rotation matrix from ECEF to NED
        R_e_n = np.zeros((self.no_meas, 3, 3))
        for i in range(self.no_meas):
            lat_rad = np.deg2rad(self.latitude[i])
            lon_rad = np.deg2rad(self.longitude[i])
            R_e_n[i,:,:] = np.array([
                [-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lon_rad), -np.cos(lat_rad)*np.cos(lon_rad)],
                [-np.sin(lat_rad)*np.sin(lon_rad),  np.cos(lon_rad), -np.sin(lon_rad)*np.cos(lat_rad)],
                [np.cos(lat_rad),                   0,               -np.sin(lat_rad)]
            ])
        self.R_e_n = R_e_n
        self.R_e_n.flags.writeable = False
    def _calc_q_en(self)                       -> None:
        # Calculate the quaternion from ECEF to NED
        q_e_n = np.zeros((self.no_meas, 4))
        for i in range(self.no_meas):
            lat_rad = np.deg2rad(self.latitude[i])
            lon_rad = np.deg2rad(self.longitude[i])

            q_lat = np.array([np.cos((lat_rad + np.pi/2)/2), 0, -np.sin((lat_rad + np.pi/2)/2), 0])
            q_lon = np.array([np.cos(lon_rad/2), 0, 0, np.sin(lon_rad/2)])

            q_v   = q_lon[0]*q_lat[1:4] + q_lat[0]*q_lon[1:4] + np.cross(q_lon[1:4], q_lat[1:4])
            q_w   = q_lon[0]*q_lat[0]   - np.transpose(q_lon[1:4]) @ q_lat[1:4]

            q = np.hstack((q_w, q_v))
            q_e_n[i,:] = q / np.linalg.norm(q)
        self.q_e_n = q_e_n
    def _euler2R_nb(self, unit='deg')          -> None:
        """
        Calculate the rotation matrix body -> ECEF from roll, pitch, yaw
        Input: roll, pitch, yaw in degrees or radians
               unit: 'deg' or 'rad' (default: 'deg') of the input angles
        Output: R_n_b: Rotation matrix body -> ECEF [no_meas x 3 x 3]
        """
        # Calculate the rotation matrix body -> ECEF from roll, pitch, yaw
        # Check if roll pitch and yaw have the same length
        R_n_b = np.zeros((self.no_meas, 3, 3))

        if unit not in ['deg', 'rad']:
            raise ValueError("unit must be either 'deg' or 'rad'")

        psi   = self.yaw - self.camera_angles['yaw']
        theta = self.pitch - self.camera_angles['pitch']
        phi   = self.roll - self.camera_angles['roll']

        if unit == 'deg':
            psi   = np.deg2rad(psi)
            theta = np.deg2rad(theta)
            phi   = np.deg2rad(phi)
        elif unit == 'rad':
            pass
        else:
            raise ValueError("unit must be either 'deg' or 'rad")

        # Calculate the roation matrix for each time step
        for i in range(self.no_meas):
            R_n_b[i,:,:] = np.array([
                               [np.cos(psi[i])*np.cos(theta[i]),  -np.sin(psi[i])*np.cos(phi[i])+np.cos(psi[i])*np.sin(theta[i])*np.sin(phi[i]),     np.sin(psi[i])*np.sin(phi[i])+np.cos(psi[i])*np.cos(phi[i])*np.sin(theta[i])],
                               [np.sin(psi[i])*np.cos(theta[i]),   np.cos(psi[i])*np.cos(phi[i])+np.sin(phi[i])*np.sin(theta[i])*np.sin(psi[i]),    -np.cos(psi[i])*np.sin(phi[i])+np.sin(theta[i])*np.sin(psi[i])*np.cos(phi[i])],
                               [-np.sin(theta[i]),                 np.cos(theta[i])*np.sin(phi[i]),                                                  np.cos(theta[i])*np.cos(phi[i])]
                               ])
            
        self.R_n_b = R_n_b
        self.R_n_b.flags.writeable = False
    def _euler2q_nb(self, quat_conv='Hamilton', unit='deg') -> None:
        """
        Convert Euler angles (roll, pitch, yaw) to a quaternion q_nb (body -> NED)
        Input:  quat_conv: 'Hamilton' q = (q_s, q_v) or 'JPL' q = (q_v, q_s)
                (default: 'Hamilton') 
                unit: 'deg' or 'rad' (default: 'deg')
        Output: q_n_b: Quaternion body -> NED [no_meas x 4]
        """
        if self.no_meas == 0 or self.yaw.any() == None or self.pitch.any() == None or self.roll.any() == None:
            raise ValueError("No measurements available")
        q_n_b = np.zeros((self.no_meas, 4))

        if quat_conv not in ['Hamilton', 'JPL']:
            raise ValueError("quat_conv must be either 'Hamilton' or 'JPL'")

        psi   = self.yaw - self.camera_angles['yaw']
        theta = self.pitch - self.camera_angles['pitch']
        phi   = self.roll - self.camera_angles['roll']

        if unit == 'deg':
            psi   = np.deg2rad(psi)
            theta = np.deg2rad(theta)
            phi   = np.deg2rad(phi)
        elif unit == 'rad':
            pass
        else:
            raise ValueError("unit must be either 'deg' or 'rad'")

        for i in range(self.no_meas):
            yaw_pitch_roll = np.array([psi[i], theta[i], phi[i]]).transpose()
            cy             = np.cos(yaw_pitch_roll * 0.5)
            sy             = np.sin(yaw_pitch_roll * 0.5)

            q = np.array([cy[0] * cy[1] * cy[2] + sy[0] * sy[1] * sy[2],
                          cy[0] * cy[1] * sy[2] - sy[0] * sy[1] * cy[2],
                          cy[0] * sy[1] * cy[2] + sy[0] * cy[1] * sy[2],
                          sy[0] * cy[1] * cy[2] - cy[0] * sy[1] * sy[2]])

            q = q / (q.transpose() @ q)
            if q[0] < 0:
                q = -q
            q_n_b[i,:] = q
        self.q_n_b = q_n_b
        self.q_n_b.flags.writeable = False
    def _R_nb2q_nb(self, quat_conv=None)      -> None:
        """
        Convert the rotation matrix R_nb (body -> NED) to a quaternion b_nb (body -> NED)
        Input:  quat_conv: 'Hamilton' q = (q_s, q_v) or 'JPL' q = (q_v, q_s)
                (default: 'Hamilton') 
        Output: q_n_b: Quaternion body -> NED [no_meas x 4]
        """
        if self.no_meas == 0 or self.R_n_b.any() == None:
            raise ValueError("No measurements available")
        if not quat_conv == None:
            if not (quat_conv == 'Hamilton' or quat_conv == 'JPL'):
                raise ValueError("quat_conv must be either 'Hamilton' or 'JPL'")
        else:    
            quat_conv = 'Hamilton'

        q_n_b = np.zeros((self.no_meas, 4))
        for i in range(self.no_meas):
            R          = self.R_n_b[i,:,:]
            q_n_b[i,:] = cc.Rot_2_quat(R, method=quat_conv)
        self.q_n_b = q_n_b
        self.q_n_b.flags.writeable = False   

    """
    OPTICAL CALCULATIONS
    """
    @staticmethod
    def _calc_fov(focal_length, sensor_dimension) -> float:
        """
        Calculate the field of view in radians
        Input: focal_length:     Focal length of the camera in [mm]
               sensor_dimension: Sensor dimension in [mm]
        Output: Field of view in radians
        """
        # Convert focal length and sensor dimensions from mm to meters
        focal_length_m     = np.float64(focal_length)     / 1000.0
        sensor_dimension_m = np.float64(sensor_dimension) / 1000.0
        # Calculate field of view in radians
        return 2 * np.arctan(sensor_dimension_m / (2 * focal_length_m))
    
    
