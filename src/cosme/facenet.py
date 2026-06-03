from __future__ import annotations

from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- [追加] projection と mediapipe.tasks をインポート ---
try:
    import projection
except ImportError:
    print("エラー: 'projection.py' が見つかりません。同じディレクトリに配置してください。")
try:
    from mediapipe import tasks
except ImportError:
     print("エラー: 'mediapipe' ライブラリが見つかりません。インストールしてください。")
# ---

# --- [追加] test.py に基づくヘルパー関数 (ここから) ---

# detect.py の上部に os がインポートされているか確認してください
# import os  <-- なければファイルの最初の方に追加

def initialize_face_landmarker():
    """MediaPipe FaceLandmarkerを初期化します。"""
    
    # detect.py があるディレクトリのパスを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 同じフォルダ内の face_landmarker.task への絶対パスを作成
    model_path = os.path.join(script_dir, 'face_landmarker.task')

    if not os.path.exists(model_path):
        print(f"エラー: モデルファイルが見つかりません。")
        print(f"探した場所: {model_path}")
        raise FileNotFoundError("face_landmarker.task が detect.py と同じフォルダにありません。")

    try:
        # 作成した絶対パス (model_path) を指定して読み込む
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options, 
            num_faces=1
        )
        detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        return detector
    except Exception as e:
        print(f"FaceLandmarkerの初期化に失敗しました。エラー詳細: {e}")
        raise

def get_uv_map(detector: mp.tasks.vision.FaceLandmarker, raw_img_rgb: np.ndarray, target_channel: np.ndarray):
    """
    画像から顔のUV展開図（指定したチャンネル）を生成します。
    
    Args:
        detector: 初期化済みの FaceLandmarker
        raw_img_rgb: 検出用のRGB画像 (H, W, 3)
        target_channel: 展開したいチャンネル (例: Gチャンネル, (H, W))

    Returns:
        UV展開図 (256, 256) または顔検出失敗時に (256, 256) のゼロ配列
    """
    
    # prepared_matrices.npz が 256x256 で作られている前提
    output_shape = (256, 256) 
    
    try:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_img_rgb)
    except Exception as e:
        print(f"MediaPipeイメージの作成に失敗: {e}")
        return np.zeros(output_shape, dtype=target_channel.dtype)

    result = detector.detect(img)
    
    if not result.face_landmarks:
        # 顔が検出されなかった場合、空のマップを返す
        # print("[Warning] UV Map: Face not detected.")
        return np.zeros(output_shape, dtype=target_channel.dtype) 

    xs = []
    ys = []
    height, width = raw_img_rgb.shape[:2]

    # ランドマーク座標を取得
    for lm in result.face_landmarks[0]:
        xs.append(lm.x * width)
        ys.append(lm.y * height)

    # 座標マッピングを計算
    pts = projection.mapping(xs, ys)
    
    # 指定されたチャンネルをマッピングに従って展開（バイリニア補間）
    uv_map = projection.inter_linear(target_channel, pts)
    
    # 形状が (256, 256) になっていることを確認
    if uv_map.shape != output_shape:
        print(f"[Warning] UV Map の形状が予期せぬ値です: {uv_map.shape}. リサイズします。")
        uv_map = cv2.resize(uv_map, output_shape, interpolation=cv2.INTER_LINEAR)

    return uv_map.astype(target_channel.dtype)

# --- [追加] ヘルパー関数 (ここまで) ---


def detect_hand(img:cv2.Mat): # 手検出
    landmark_line_ids = [ 
        (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 掌
        (1, 2), (2, 3), (3, 4),         # 親指
        (5, 6), (6, 7), (7, 8),         # 人差し指
        (9, 10), (10, 11), (11, 12),    # 中指
        (13, 14), (14, 15), (15, 16),   # 薬指
        (17, 18), (18, 19), (19, 20),   # 小指
    ]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,                # 最大検出数
        min_detection_confidence=0.5,   # 検出信頼度
        min_tracking_confidence=0.5     # 追跡信頼度
    )
    _img = img.copy()
    img_h, img_w, _ = _img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    lm_c = (64, 0, 0)
    # 検出処理の実行
    results = hands.process(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB))
    
    pt_1 = None # 検出失敗時に備えて初期化

    if results.multi_hand_landmarks:
        # 検出した手の数分繰り返し
        for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # landmarkの繋がりをlineで表示
            for line_id in landmark_line_ids:
                # 1点目座標取得
                lm = hand_landmarks.landmark[line_id[0]]
                lm_pos1 = (int(lm.x * img_w), int(lm.y * img_h))
                # 2点目座標取得
                lm = hand_landmarks.landmark[line_id[1]]
                lm_pos2 = (int(lm.x * img_w), int(lm.y * img_h))
                # line描画
                cv2.line(_img, lm_pos1, lm_pos2, (128, 0, 0), 1)

            # landmarkをcircleで表示
            z_list = [lm.z for lm in hand_landmarks.landmark]
            z_min = min(z_list)
            z_max = max(z_list)
            for lm_id, lm in enumerate(hand_landmarks.landmark):
                lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
                lm_z = int((lm.z - z_min) / (z_max - z_min) * 255)
                cv2.circle(_img, lm_pos, 3, (255, lm_z, lm_z), -1)
                cv2.putText(_img, str(lm_id), lm_pos, font, 1.0, lm_c, 1)
            
            pt_1 = draw_chosen_pt(img, hand_landmarks, 13)
            if pt_1: # pt_1 が None でないことを確認
                square_1 = get_square_region(pt_1)
                _img[square_1[0][1]:square_1[1][1], square_1[0][0]:square_1[1][0]] = (0, 255, 0)  

            # 検出情報をテキスト出力
            # - テキスト情報を作成
            hand_texts = []
            for c_id, hand_class in enumerate(results.multi_handedness[h_id].classification):
                hand_texts.append("#%d-%d" % (h_id, c_id)) 
                hand_texts.append("- Index:%d" % (hand_class.index))
                hand_texts.append("- Label:%s" % (hand_class.label))
                hand_texts.append("- Score:%3.2f" % (hand_class.score * 100))
            # - テキスト表示に必要な座標など準備
            lm = hand_landmarks.landmark[0]
            lm_x = int(lm.x * img_w) - 50
            lm_y = int(lm.y * img_h) - 10
        
            # - テキスト出力
            for cnt, text in enumerate(hand_texts):
                cv2.putText(_img, text, (lm_x, lm_y + 10 * cnt), font, 0.3, lm_c, 1)
        
    hands.close() # インスタンスを終了
    return pt_1


def detect_face(img:cv2.Mat): # 顔検出
    mp_drawing = mp.solutions.drawing_utils # 描画用のインスタンス
    mp_face_mesh = mp.solutions.face_mesh # MLソリューションの顔メッシュインスタンス

    face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5)

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cropped_img = img.copy() # 描画用の画像をコピーしておく
    results = face_mesh.process(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)) # 顔メッシュを計算

    pt_2 = None # 検出失敗時に備えて初期化

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks: # 画像内の全ての顔の顔特徴点
         
            mp_drawing.draw_landmarks(
            image=cropped_img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec) # 特徴点の描画

            for i, landmark in enumerate(face_landmarks.landmark):
                    h, w, _ = cropped_img.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)  
                    cv2.putText(cropped_img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            pt_2 = draw_chosen_pt(cropped_img, face_landmarks, 101)
            # square_2 = get_square_region(pt_2)
            # cropped_img[square_2[0][1]:square_2[1][1], square_2[0][0]:square_2[1][0]] = (0, 255, 0)

    face_mesh.close() # インスタンスを終了させる
    return  pt_2

def draw_chosen_pt(_img:cv2.Mat, landmarks, id:int): # 薬指の付け根、目の下の座標取得
    # hand 13
    # face 101
    
    if not landmarks:
        return None

    img_h, img_w, _ = _img.shape
    
    if id >= len(landmarks.landmark):
        print(f"要求されたlandmark ID {id} が範囲外です。")
        return None

    lm_id = landmarks.landmark[id]
    lm_id_pos = (int(lm_id.x * img_w), int(lm_id.y * img_h))
            
    cv2.putText(_img, f"id: ({lm_id_pos[0]}, {lm_id_pos[1]})", 
                        (lm_id_pos[0] + 10, lm_id_pos[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
    
    return lm_id_pos

def get_square_region(pt:tuple): # 座標から範囲指定
    if pt is None:
        # デフォルトの領域を返すかエラーを出す
        return (0, 0), (20, 20) 
    center_x, center_y = pt
    side_length = 20
    top_left = (center_x - side_length // 2, center_y - side_length // 2)
    bottom_right = (center_x + side_length // 2, center_y + side_length // 2)

    return top_left, bottom_right

def get_sample_pos_signal(img_dir:Path, start_f:int, end_f:int, fromface:bool): #1フレーム目のみ検出
    print(str(img_dir))
    img_paths = img_dir.glob("*.npy") 
    img_paths = [str(p) for p in img_paths]

    sample_signal = []
    
    pt = None
    square = None
    accum_face = None
    face_low_cut = None

    for i, img_path in tqdm(enumerate(sorted(img_paths)), total=len(img_paths[:end_f])): 
        img = np.load(img_path)

        if i == 0:
            if fromface: #顔からリファレンスベクトルを求める場合
                pt = detect_face(img)   
            else:
                pt = detect_hand(img)
            
            if pt is None:
                raise ValueError("顔または手の検出に失敗しました。")
            
            square = get_square_region(pt)
            face = img[square[0][1]:square[1][1], square[0][0]:square[1][0], 1]
            if face.size == 0:
                raise ValueError("参照領域の切り出しに失敗しました。")

            accum_face = face.astype(np.float64)
            face_low_cut = face.astype(np.float64)
        
        else:
            if square is None:
                raise ValueError("参照領域(square)が初期化されていません。")
            face = img[square[0][1]:square[1][1], square[0][0]:square[1][0], 1]
            if face.size == 0:
                print(f"フレーム {i} で空の領域が検出されました。スキップします。")
                continue

            accum_face = accum_face * 0.95 + face * 0.05
            face_low_cut = face - accum_face

        if start_f <= i+1 <= end_f+1:
            sample_signal.append(np.mean(face_low_cut))
        elif i+1 > end_f+1:
            break

    # --- グラフ描画 ---
    fig, ax = plt.subplots()
    ax.plot(sample_signal, color='green')
    ax.set_yticks([])
    plt.show()
    # ---

    return np.array(sample_signal)


# (毎フレーム検出する get_sample_pos_signal はコメントアウトされたまま)


def visualize_inner(img_dir: Path, base_phase:np.ndarray, start_f:int, end_f:int, fromface:bool, save_flag:bool=False): 
    
    # --- [修正] Landmarkerを初期化 ---
    try:
        detector = initialize_face_landmarker()
    except Exception:
        return None # 初期化失敗時は処理終了
    # ---

    img_paths = img_dir.glob("*.npy")
    img_paths = [str(p) for p in img_paths]

    raw_green_signal = []
    filtered_green_signal = []

    first_img = None
    accum_face = None
    face_low_cut = None
    inner = None

    for i, img_path in tqdm(enumerate(sorted(img_paths)), total=len(img_paths[:end_f])): 
        img = np.load(img_path)
        
        # --- [修正] ここで展開図を生成 ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Gチャンネルを展開図にする (float64で)
        uv_g_channel = get_uv_map(detector, img_rgb, img[..., 1].astype(np.float64))
        # ---

        if i == 0:
            first_img = img[:,:,:]
            
            # --- [修正] 処理対象を展開図に差し替え ---
            accum_face = uv_g_channel # (256, 256)
            face_low_cut = uv_g_channel # (256, 256)
            inner = np.zeros_like(accum_face, dtype=np.float64) # (256, 256)
            # ---

            # --- 初回フレームのデータをリストに追加 ---
            if start_f <= i+1 <= end_f+1:
                raw_green_signal.append(np.mean(accum_face)) # 展開図の平均
                filtered_green_signal.append(np.mean(face_low_cut)) # 展開図の平均
            # --- ここまで ---
        else:
            # --- [修正] 処理対象を展開図に差し替え ---
            face = uv_g_channel # (256, 256)
            # ---

            accum_face = accum_face * 0.95 + face * 0.05
            face_low_cut = face - accum_face

        if start_f <= i+1 <= end_f+1:
            if i+1-start_f < len(base_phase): # base_phase のインデックスチェック
                inner += face_low_cut * base_phase[i+1-start_f]
            # --- プロット用のデータをリストに追加 ---
            raw_green_signal.append(np.mean(face))
            filtered_green_signal.append(np.mean(face_low_cut))
            # --- ここまで ---
            # print(len(inner)) # (デバッグ用)
        elif i+1 > end_f+1:
            break

    # --- [修正] detector を閉じる ---
    detector.close()
    # ---

    if first_img is None:
        print("画像が処理されませんでした。")
        return None
        
    # --- プロット処理 ---
    fig1, ax1 = plt.subplots()
    ax1.plot(raw_green_signal, color='green')
    ax1.set_title("Raw Green Component Fluctuation (UV Map Avg)")
    ax1.set_yticks([])
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(filtered_green_signal, color='green')
    ax2.set_title("Filtered Green Component Fluctuation (UV Map Avg)")
    ax2.set_yticks([])
    plt.show()
    # ---

    # --- 可視化 ---
    # (参照領域の取得ロジック - 元画像に対して)
    pt = None
    if fromface: #顔からリファレンスベクトルを求める場合
            pt = detect_face(first_img) # 初回画像で検出
    else:
            pt = detect_hand(first_img) # 初回画像で検出
    square = get_square_region(pt)

    eye_coords = get_eye_landmarks(first_img)
    img_with_lines = first_img.copy()
    line_thickness = 40
    if eye_coords:
        line_color = (0, 0, 0)
        cv2.line(img_with_lines, eye_coords["left_outer"], eye_coords["right_outer"], line_color, line_thickness)

    fig = plt.figure(figsize=(10, 5), dpi=100)
    
    # --- [修正] ax1 (元画像) はそのまま ---
    # ax1 = fig.add_subplot(121)
    # ax1.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    # ... (ax1 の xlim/ylim設定) ...
    # rect = plt.Rectangle(...) # 参照領域の矩形
    # ax1.add_patch(rect)
    # ax1.axis("off")

    # --- [修正] ax2 (内積結果) は (256, 256) の展開図を表示 ---
    ax2 = fig.add_subplot(122) # (もしax1も表示するなら 122, ax2だけなら 111)
    gamma = 0.25
    ax2.imshow((np.clip(inner, 0, 255)/255.0)**gamma, vmax=None, vmin=0)
    
    # inner の形状 (256, 256) に基づいてマージンを設定
    inner_height, inner_width = inner.shape[:2]
    cut_margin2_x = int(inner_width * 0)#0.24
    cut_margin2_y = int(inner_height * 0)#0.18
    ax2.set_xlim(cut_margin2_x, inner_width - cut_margin2_x)
    ax2.set_ylim(inner_height - cut_margin2_y, 0)

    # (目線は座標系が違うためコメントアウト)
    # if eye_coords:
    #     ax2.plot(...)

    ax2.axis("off")
    plt.subplots_adjust(wspace=0)
    
    # ---
    
    if save_flag == True:
        output_dir = img_dir.parent / "output_"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_name = img_dir.name
        fig.savefig(
            f"{output_dir}/{output_name}_abs.pdf", bbox_inches="tight", pad_inches=0.0
        )
        plt.close()
    else:
        plt.show()
        
    return inner
    ...


def plot_pixel_signals(img_dir: Path, start_f: int, end_f: int):
    """
    顔の特定1ピクセルの緑色成分の変動をプロットする専用の関数。
    - 生の緑色成分の変動
    - フィルターをかけた後の緑色成分の変動
    の2つの波形をグラフで表示します。
    (この関数は UV展開 を *使用しません* - 元のロジックのままです)
    """
    img_paths = sorted(img_dir.glob("*.npy"))
    if not img_paths:
        raise ValueError(f"画像ファイルが見つかりません: {img_dir}")
    img_paths = [str(p) for p in img_paths]

    raw_pixel_signal = []
    filtered_pixel_signal = []
    
    target_px, target_py = None, None
    
    first_img_for_detect = np.load(img_paths[0])
    h, w, _ = first_img_for_detect.shape
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    results = face_mesh.process(cv2.cvtColor(first_img_for_detect, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmark = face_landmarks.landmark[101] # 目の下のランドマーク
        target_px = int(landmark.x * w)
        target_py = int(landmark.y * h)
        print(f"追跡するピクセルを特定しました: ({target_px}, {target_py})")
    else:
        face_mesh.close()
        raise ValueError("最初のフレームで顔を検出できませんでした。ピクセルを特定できません。")
    face_mesh.close()

    accum_face = None
    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        if i > end_f:
            break

        img = np.load(img_path)
        face = img[..., 1].astype(np.float64) # Gチャンネル

        if accum_face is None:
            accum_face = face
        
        accum_face = accum_face * 0.95 + face * 0.05
        face_low_cut = face - accum_face

        if start_f <= i < end_f:
            if target_py < face.shape[0] and target_px < face.shape[1]:
                raw_pixel_signal.append(face[target_py, target_px])
                filtered_pixel_signal.append(face_low_cut[target_py, target_px])
            else:
                print(f"警告: ピクセル ({target_px}, {target_py}) がフレーム {i} の範囲外です。")

    # --- プロット ---
    fig1, ax1 = plt.subplots()
    ax1.plot(raw_pixel_signal, color='green')
    ax1.set_ylim(0, 255)
    ax1.set_yticks([])
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(filtered_pixel_signal, color='green')
    ax2.set_yticks([])
    plt.show()

    return



def get_eye_landmarks(img: np.ndarray): #目線処理用

    # print(f"[DEBUG] Eye detection - Input shape: {img.shape}, dtype: {img.dtype}")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )

    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    coords = None
    if results.multi_face_landmarks:
        # print("[DEBUG] Face landmarks found.")
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = img.shape
            p_right_outer = face_landmarks.landmark[33]
            p_right_inner = face_landmarks.landmark[133]
            p_left_inner = face_landmarks.landmark[362]
            p_left_outer = face_landmarks.landmark[263]

            coords = {
                "right_outer": (int(p_right_outer.x * w), int(p_right_outer.y * h)),
                "right_inner": (int(p_right_inner.x * w), int(p_right_inner.y * h)),
                "left_inner":  (int(p_left_inner.x * w), int(p_left_inner.y * h)),
                "left_outer":  (int(p_left_outer.x * w), int(p_left_outer.y * h)),
            }
            break
    else:
        print("[WARNING] Face landmarks NOT found.")

    face_mesh.close()
    return coords

def visualize_inner_for_pdf(img_dir: Path, base_phase:np.ndarray, start_f:int, end_f:int, fromface:bool): #可視化一覧用
    
    # --- [修正] Landmarkerを初期化 ---
    try:
        detector = initialize_face_landmarker()
    except Exception:
        return None, None # 初期化失敗
    # ---
    
    img_paths = sorted(img_dir.glob("*.npy"))
    img_paths = [str(p) for p in img_paths]

    first_img = None
    accum_face = None
    inner = None

    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths[:end_f+2])):
        if i > end_f + 1:
            break
        
        img = np.load(img_path)
        
        # --- [修正] 展開図を生成 ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        uv_g_channel = get_uv_map(detector, img_rgb, img[..., 1].astype(np.float64))
        # ---

        if i == 0:
            first_img = img.copy()
            accum_face = uv_g_channel # (256, 256)
            inner = np.zeros_like(accum_face, dtype=np.float64) # (256, 256)
            continue
        
        # (フォールバック)
        if first_img is None:
             first_img = img.copy()
             accum_face = uv_g_channel
             inner = np.zeros_like(accum_face, dtype=np.float64)

        face = uv_g_channel # (256, 256)
        accum_face = accum_face * 0.95 + face * 0.05
        face_low_cut = face - accum_face

        if start_f <= i+1 <= end_f+1:
             if i+1-start_f < len(base_phase): # インデックスチェック
                inner += face_low_cut * base_phase[i+1-start_f]

    # --- [修正] detector を閉じる ---
    detector.close()
    # ---

    if first_img is None:
        print(f"[ERROR] No images were processed for {img_dir.name}.")
        return None, None

    eye_coords = get_eye_landmarks(first_img)
    
    img_with_lines = first_img.copy()
    line_thickness = 45

    if eye_coords:
        line_color = (0, 0, 0)
        cv2.line(img_with_lines, eye_coords["left_outer"], eye_coords["right_outer"], line_color, line_thickness)

    fig = plt.figure(figsize=(10, 5), dpi=100)

    ax1 = fig.add_subplot(121)
    ax1.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    ax1.axis("off")

    ax2 = fig.add_subplot(122)
    gamma = 0.25
    ax2.imshow((np.clip(inner, 0, 255)/255.0)**gamma,vmax=None, vmin=0) # (256, 256)
    ax2.axis("off")

    # --- [修正] ax2 の目線描画は座標系が異なるためコメントアウト ---
    # if eye_coords:
    #     ax2.plot(
    #         [eye_coords["left_outer"][0], eye_coords["right_outer"][0]],
    #         [eye_coords["left_outer"][1], eye_coords["right_outer"][1]],
    #         color='black',
    #         linewidth=17
    #     )
    # ---

    margin_left_ratio = 0.22 #各図の左右を削る
    margin_right_ratio = 0.12

    # --- [修正] ax1 と ax2 でそれぞれの幅を基準に xlim を設定 ---
    img_width = first_img.shape[1]
    x_min1 = int(img_width * margin_left_ratio)
    x_max1 = int(img_width * (1 - margin_right_ratio))
    ax1.set_xlim(x_min1, x_max1)
    
    inner_width = inner.shape[1] # 256
    x_min2 = int(inner_width * margin_left_ratio)
    x_max2 = int(inner_width * (1 - margin_right_ratio))
    ax2.set_xlim(x_min2, x_max2)
    # ---

    plt.subplots_adjust(wspace=0, hspace=0)

    return fig, inner


# (create_tracking_video はコメントアウトされたまま)


# (if __name__ == "__main__": の部分は元のまま)
if __name__ == "__main__":
    # このスクリプトを直接実行する場合のロジック
    # (例: パスを指定して実行)
    # このサンプルでは、'img_dir' や 'base_phase' などが未定義のため、
    # このままでは実行できません。
    # 呼び出し元のスクリプトでこれらの関数を使ってください。
    print("detect.py がモジュールとしてロードされました。")
    print("関数 (visualize_inner など) を外部から呼び出して使用してください。")
    
    # 例:
    # output_dir = Path("some_output_directory")
    # img_dir = Path("some_image_directory")
    # output_dir.mkdir(exist_ok=True)
    # ... proc(...) の呼び出し ... (proc関数はここには定義されていませんが)