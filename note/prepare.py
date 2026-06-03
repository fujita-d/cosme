''' Prepare transformation matrices.
Run this script once at first or when output image size need to be changed.'''

### BEGIN: config ###

output_width = 256
output_height = 256

### END: config ###


# Imports
import numpy as np
from matplotlib import pyplot
import tqdm
import os  # <-- [追加] osモジュールをインポート

# Read FaceMesh .obj file and convert it to several forms for usage.
# (only specific version of .obj file can be used)

# --- [ここから修正] ---
# スクリプトが置かれているディレクトリの絶対パスを取得
script_dir = os.path.dirname(os.path.abspath(__file__))
# .objファイルへの絶対パスを作成
obj_file_path = os.path.join(script_dir, 'canonical_face_model.obj')

try:
    # 修正した絶対パスでファイルを開く
    with open(obj_file_path) as f:
        dt = f.read().split('\n')
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません。")
    print(f"次のパスに 'canonical_face_model.obj' が存在するか確認してください: \n{obj_file_path}")
    exit() # ファイルがなければ処理を終了
# --- [ここまで修正] ---

vs = [v.split()[1:] for v in dt[:468]]
vts = [vt.split()[1:] for vt in dt[468:936]]
fs = [f.split()[1:] for f in dt[936:1834]]

v_arr = np.array([(float(x), float(y), float(z)) for x, y, z in vs])
vt_arr = np.array([(float(u), float(v)) for u, v in vts])

polygon_ver_indices = [(int(f[0].split('/')[0]) - 1,
                        int(f[1].split('/')[0]) - 1,
                        int(f[2].split('/')[0]) - 1)
                       for f in fs]

polygon_tex_indices = [(int(f[0].split('/')[1]) - 1,
                        int(f[1].split('/')[1]) - 1,
                        int(f[2].split('/')[1]) - 1)
                       for f in fs]

# Define functions for polygon matching.
# (For each output pixels, search a polygon which contains the pixel.)
def tex_bounding_box(tex_indices):
    ''' calculate bounding box. (triangle polygon => bounding box) '''
    i1, i2, i3 = tex_indices
    pt1, pt2, pt3 = vt_arr[i1], vt_arr[i2], vt_arr[i3]
    u, v = list(zip(pt1, pt2, pt3))
    return (max(u), min(u), max(v), min(v))

tex_polygon_bb = [tex_bounding_box(pti) for pti in polygon_tex_indices]
tex_polygon = [(vt_arr[i1], vt_arr[i2], vt_arr[i3]) for i1, i2, i3 in polygon_tex_indices]

def check_inside_bb(pt, bb):
    ''' Roughly screening. (point, bounding box) => Bool '''
    u, v = pt
    umax, umin, vmax, vmin = bb
    if u < umin: return False
    if u > umax: return False
    if v < vmin: return False
    if v > vmax: return False
    return True

def angle_imaginary(v1, v2):
    ''' Calculate angle of two vectors using complex number
        and return imaginary part only.'''
    return v1[1] * v2[0] - v1[0] * v2[1]

def vec(pt1, pt2):
    ''' Return a vector of pt1 -> pt2 '''
    u1, v1 = pt1
    u2, v2 = pt2
    return (u2 - u1, v2 - v1)

def check_inside_polygon(pt, polygon):
    ''' Final checking. (point, polygon) => Bool '''
    pt1, pt2, pt3 = polygon
    a1 = angle_imaginary(vec(pt1, pt2), vec(pt1, pt))
    a2 = angle_imaginary(vec(pt2, pt3), vec(pt2, pt))
    a3 = angle_imaginary(vec(pt3, pt1), vec(pt3, pt))
    if a1*a2 > 0 and a2*a3 > 0 and a3*a1 > 0: return True
    return False

def search_polygon(u, v):
    ''' Search a polygon which contains the pixel.
        pos(u, v) => polygon index[0-898) or unhit(-1)'''
    for i in range(898):
        if not check_inside_bb((u, v), tex_polygon_bb[i]): continue
        if check_inside_polygon((u, v), tex_polygon[i]): return i
    return -1

# Define function for blending weight generation.
def calc_weights(pt, pt1, pt2, pt3):
    ''' Describe pt as weighted average of three points. '''
    a = np.array(
        [[pt1[0], pt2[0], pt3[0]],
         [pt1[1], pt2[1], pt3[1]],
         [1., 1., 1.]])
    b = np.array([pt[0], pt[1], 1.])
    return np.linalg.solve(a, b)

# Generate blank matrices.
def g_pt_weight_arr(): return np.zeros((output_height, output_width))
def g_pt_index_arr(): return np.zeros((output_height, output_width), dtype=int)

pt1_arr, pt1v_arr = g_pt_weight_arr(), g_pt_index_arr()
pt2_arr, pt2v_arr = g_pt_weight_arr(), g_pt_index_arr()
pt3_arr, pt3v_arr = g_pt_weight_arr(), g_pt_index_arr()

# Calculate 3-point indices and weights in each points.
for j in tqdm.tqdm(range(output_height)):
    for i in range(output_width):
        u, v = i / output_width, j / output_height
        poly_index = search_polygon(u, v)
        if poly_index < 0: continue

        # Weights are calculated in UV (texture) space.
        pt1, pt2, pt3 = tex_polygon[poly_index]
        w1, w2, w3 = calc_weights((u, v), pt1, pt2, pt3)

        # Given vertices are in 3D (vertex) space.
        pt1v, pt2v, pt3v = polygon_ver_indices[poly_index]

        pt1_arr[j, i], pt1v_arr[j, i] = w1, pt1v
        pt2_arr[j, i], pt2v_arr[j, i] = w2, pt2v
        pt3_arr[j, i], pt3v_arr[j, i] = w3, pt3v

# Save
np.savez('prepared_matrices.npz',
         pt1_w = pt1_arr, pt1_v = pt1v_arr,
         pt2_w = pt2_arr, pt2_v = pt2v_arr,
         pt3_w = pt3_arr, pt3_v = pt3v_arr)
         
