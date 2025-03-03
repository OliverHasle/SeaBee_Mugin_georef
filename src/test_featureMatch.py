import os
#import tools.visualizations as vis

from __init__                 import initialize
from objects.FeatureMatching  import FeatureMatching

def main():
    # Example usage
    config, parameter = initialize()
    featureMatch      = FeatureMatching(config)
    featureMatch.process_image_grid()
    print("Feature Matching Done")

if __name__ == "__main__":
    main()
