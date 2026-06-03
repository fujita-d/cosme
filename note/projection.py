''' Face projection into UV-Texture space using face landmarks from mediapipe. '''

# Imports
import numpy as np
import os

# Load prepared matrices.
script_dir = os.path.dirname(os.path.abspath(__file__))
npz_path = os.path.join(script_dir, 'prepared_matrices.npz')

matrices = np.load(npz_path) 
pt1_w, pt1_v = matrices['pt1_w'], matrices['pt1_v']
pt2_w, pt2_v = matrices['pt2_w'], matrices['pt2_v']
pt3_w, pt3_v = matrices['pt3_w'], matrices['pt3_v']

def mapping(ver_x, ver_y):
    ''' UV Map in image space. '''
    pt1x = np.take(ver_x, pt1_v)
    pt1y = np.take(ver_y, pt1_v)
    pt2x = np.take(ver_x, pt2_v)
    pt2y = np.take(ver_y, pt2_v)
    pt3x = np.take(ver_x, pt3_v)
    pt3y = np.take(ver_y, pt3_v)

    ptx = pt1x * pt1_w + pt2x * pt2_w + pt3x * pt3_w
    pty = pt1y * pt1_w + pt2y * pt2_w + pt3y * pt3_w

    return (ptx, pty)

def inter_nearest(src_img, pts):
    ''' Nearest-Neighbor interpolation. '''
    ptx, pty = pts
    pos = (np.array(np.round(ptx), dtype=int) +
           np.array(np.round(pty), dtype=int) * src_img.shape[1])
    return np.take(src_img.flatten(), pos)

def inter_linear(src_img, pts):
    ''' Bilinear interpolation. '''
    ptx, pty = pts
    ptx_f, ptx_i = np.modf(ptx)
    pty_f, pty_i = np.modf(pty)

    ptx_floor = np.array(np.floor(ptx_i), dtype=int)
    pty_floor = np.array(np.floor(pty_i), dtype=int)
    
    lw = src_img.shape[1]
    
    pt_l_t = pty_floor * lw + ptx_floor
    pt_l_b = pt_l_t + lw
    pt_r_t, pt_r_b = pt_l_t + 1, pt_l_b + 1

    flatten_img = src_img.flatten()
    def v(p): return np.take(flatten_img, p)

    return ((v(pt_l_t) * (1 - ptx_f) + v(pt_r_t) * ptx_f) * (1 - pty_f) +
            (v(pt_l_b) * (1 - ptx_f) + v(pt_r_b) * ptx_f) * pty_f)
             
