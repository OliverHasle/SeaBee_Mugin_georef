import numpy as np

def Rot_2_quat(R, method='Hamilton'):
    """Converts a rotation matrix to a quaternion.
    Args:
        R:      3x3 rotation matrix
        method: 'Hamilton' or 'JPL' quaternion convention (default is 'Hamilton')
    Returns:
        q: 4x1 quaternion
    """
    if method not in ['Hamilton', 'JPL']:
        raise ValueError("quat_conv must be either 'Hamilton' or 'JPL'")
    diag_elements = np.diagonal(R)
    R_trace       = np.trace(R)
    k             = np.argmax([diag_elements[0], diag_elements[1], diag_elements[2], R_trace])
    if k == 0:  # R[0, 0] is largest
        p_k = np.sqrt(1 + 2 * R[0, 0] - R_trace)
        q_x = p_k
        q_y = (R[1, 0] + R[0, 1]) / p_k
        q_z = (R[0, 2] + R[2, 0]) / p_k
        q_w = (R[2, 1] - R[1, 2]) / p_k
    elif k == 1:  # R[1, 1] is largest
        p_k = np.sqrt(1 + 2 * R[1, 1] - R_trace)
        q_x = (R[1, 0] + R[0, 1]) / p_k
        q_y = p_k
        q_z = (R[2, 1] + R[1, 2]) / p_k
        q_w = (R[0, 2] - R[2, 0]) / p_k
    elif k == 2:  # R[2, 2] is largest
        p_k = np.sqrt(1 + 2 * R[2, 2] - R_trace)
        q_x = (R[0, 2] + R[2, 0]) / p_k
        q_y = (R[2, 1] + R[1, 2]) / p_k
        q_z = p_k
        q_w = (R[1, 0] - R[0, 1]) / p_k
    else:  # trace is largest
        p_k = np.sqrt(1 + R_trace)
        q_x = (R[2, 1] - R[1, 2]) / p_k
        q_y = (R[0, 2] - R[2, 0]) / p_k
        q_z = (R[1, 0] - R[0, 1]) / p_k
        q_w = p_k
    if method == 'Hamilton':
        q = 0.5*np.array([ q_w, q_x, q_y, q_z])
    elif method == 'JPL':
        q = 0.5*np.array([ q_x, q_y, q_z, q_w])
    else:
        raise ValueError("method must be either 'Hamilton' or 'JPL'")
            
    q = q / (q.transpose() @ q)
    if q[0] < 0:
        q = -q
    return q