import os
import multiprocessing as mp
#from functools

from __init__                 import initialize
from objects.Orthorectification  import Orthorectification

def main():
    # Example usage
    config, parameter = initialize()
    featureMatch      = Orthorectification(config)
    featureMatch.main_orthorectify_images()
    print("Feature Matching Done")

def main_multithreaded():
    # Example usage
    config, parameter = initialize()
    featureMatch      = Orthorectification(config)

    # Multithreading
    num_threads = 4
    pool        = mp.Pool(num_threads)
    pool.map(featureMatch.main_orthorectify_images, range(num_threads), verbose=False)
    pool.close()

if __name__ == "__main__":
    main()
#    main_multithreaded()
