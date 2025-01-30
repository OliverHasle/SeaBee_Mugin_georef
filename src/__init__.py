import json               as json
import configparser       as cp
import numpy              as np
import xmltodict          as xd

__version__ = "0.1.0"

def initialize():
    config = cp.ConfigParser()
    config.read('CONFIG\\config.ini')

    # Read the XML file
    with open(config['SETTINGS']['parameters']) as file:
        param = xd.parse(file.read())

    for sens in param['config']['sensor']:
        if sens['name'] == config['SETTINGS']['sensor']:
            sensor = sens
            break

    if 'p_bc_b' in param['config']['sensor_positions']:
        p_bc_b = np.array([float(param['config']['sensor_positions']['p_bc_b']['x']),
                           float(param['config']['sensor_positions']['p_bc_b']['y']),
                           float(param['config']['sensor_positions']['p_bc_b']['z'])])
    else:
        print("No lever arm \"drone -> sensor\" found in the configuration file. Setting lever arm to zero.")
        p_bc_b = np.array([0, 0, 0])

#    if 'R_imu_b' in param['config']['sensor_positions']:
#        R_imu_b_str = param['config']['sensor_positions']['R_imu_b']
#        R_imu_b     = _parse_matrix_string(R_imu_b_str)
#    else:
#        print("No rotation matrix \"IMU -> body\" found in the configuration file. Setting rotation matrix to identity.")
#        R_imu_b = np.eye(3)

    if 'camera_angles' in param['config']['sensor_positions']:
        roll  = float(param['config']['sensor_positions']['camera_angles']['roll'])
        pitch = float(param['config']['sensor_positions']['camera_angles']['pitch'])
        yaw   = float(param['config']['sensor_positions']['camera_angles']['yaw'])
        camera_angles = {
            'roll':  roll,
            'pitch': pitch,
            'yaw':   yaw
        }
    else:
        print("No camera angles found in the configuration file. Setting camera angles to zero.")
        camera_angles = {
            'roll':  0,
            'pitch': 0,
            'yaw':   0
        }

    if 'l_cd' in param['config']['max_ray_length']:
        l_cd = np.array(float(param['config']['max_ray_length']['l_cd']))
    else:
        print("No maximum ray length found in the configuration file. Setting maximum ray length to 1000 meters.")
        l_cd = 1000
    
    # Write into a dictionary
    parameter = {
        'sensor':        sensor,
        'p_bc_b':        p_bc_b,
        'l_cd':          l_cd,
        'camera_angles': camera_angles
    }

    return config, parameter, 

def _parse_matrix_string(matrix_str):
    rows = []
    for row in matrix_str['Row']:
        # Split and convert to float
        row_values = [float(x.strip()) for x in row.split(',')]
        rows.append(row_values)
   
    return np.array(rows)

# Export the initialize function
__all__ = ['initialize']
#__all__ = ['initialize', 'optical_calc', 'georeferencing']