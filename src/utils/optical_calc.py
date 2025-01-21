import numpy as np

def calc_fov(focal_length, sensor_dimension):
        # Convert focal length and sensor dimensions from mm to meters
    focal_length_m     = focal_length / 1000.0
    sensor_dimension_m = sensor_dimension / 1000.0
    # Calculate field of view in radians
    return 2 * np.arctan(sensor_dimension_m / (2 * focal_length_m))

def get_geotagging(img):
    exif = img._getexif()
    if exif is not None:
        geotagging = {}
        for (idx, tag) in TAGS.items():
            if tag == 'GPSInfo':
                if idx not in exif:
                    raise ValueError("No EXIF geotagging found")
                for (t, value) in GPSTAGS.items():
                    if t in exif[idx]:
                        geotagging[value] = exif[idx][t]
        return geotagging
    else:
        raise ValueError("No EXIF metadata found")