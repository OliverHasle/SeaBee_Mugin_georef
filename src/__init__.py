from utils import optical_calc, georeferencing

import json               as json
import configparser       as cp
import numpy              as np

__version__ = "0.1.0"

def initialize():
    config = cp.ConfigParser()
    config.read('misc\\config.ini')

    # pack everything in a dict and return
    init_param = {
        'mission_name': config['MISSION']['missionName'],
        'inputfolder':  config['MISSION']['inputfolder'],
        'outputfolder': config ['MISSION']['outputfolder'],
        'dem_path':     config['ELEVATION MODELS']['dem_path'],
        'model_path':   config['ELEVATION MODELS']['model_path'],
        'sensor':       json.loads(config['SENSOR']['Sony ILX-LR1']),
        'p_bc_b':       np.array([float(x) for x in config['LEVERARMS']['p_bc_b'].strip('()').split(',')])
    }
    return init_param, config

# Export the initialize function
__all__ = ['initialize']
#__all__ = ['initialize', 'optical_calc', 'georeferencing']