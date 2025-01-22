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

    #for leverArm in param['config']['lever_arm']:
    
    if 'p_bc_b' in param['config']['lever_arm']:
        p_bc_b = np.array([float(param['config']['lever_arm']['p_bc_b']['x']),
                       float(param['config']['lever_arm']['p_bc_b']['y']),
                       float(param['config']['lever_arm']['p_bc_b']['z'])])
    else:
        print("No lever arm \"drone -> sensor\" found in the configuration file. Setting lever arm to zero.")
        p_bc_b = np.array([0, 0, 0])

    # Write into a dictionary
    parameter = {
        'sensor': sensor,
        'p_bc_b': p_bc_b
    }

    return config, parameter, 

# Export the initialize function
__all__ = ['initialize']
#__all__ = ['initialize', 'optical_calc', 'georeferencing']