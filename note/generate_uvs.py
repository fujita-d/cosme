import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from tqdm import tqdm
import os
import projection  # 同階層のprojection.py

# --- 設定項目 ------------------------------------------------
# 素肌時の .npy ファイルが入っているディレクトリ
BARE_FACE_DIR = r"C:\Users\hil\Documents\vimba_rgb_cam\20260325\exp004_20260325_14_22_bare0_p70"

# 化粧時の .npy ファイルが入っているディレクトリ
MAKEUP_DIR = r"C:\Users\hil\Documents\vimba_rgb_cam\20260325\exp004_20260325_14_28_fund1_2_p70"

# 出力先ディレクトリ名
OUTPUT_ROOT = "output_uvs_texture"
# ------------------------------------------------------------

def initialize_face_landmarker():
    """MediaPipe FaceLandmarkerを初期化します（バイナリ読み込み版）。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    paths_to_check = [
        os.path.join(script_dir, 'face_landmarker.task'),
        'face_landmarker.task'
    ]
    
    model_path = None
    for p in paths_to_check:
        if os.path.exists(p):
            model_path = p
            break
            
    if model_path is None:
        raise FileNotFoundError("face_landmarker.task が見つかりません。")

    with open(model_path, 'rb') as f:
        model_buffer = f.read()

    base_options = python.BaseOptions(model_asset_buffer=model_buffer)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)

def generate_uv_from_npy(detector, npy_img):
    """
    1フレーム分のnpy画像データ(BGR)を受け取り、UV展開図を生成します。
    """
    if npy_img.dtype != np.uint8:
        det_img = cv2.normalize(npy_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(npy_img, cv2.COLOR_BGR2RGB)

    height, width = img_rgb.shape[:2]
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        return None # 顔が見つからない場合

    # ランドマーク座標の取得
    xs = []
    ys = []
    for lm in detection_result.face_landmarks[0]:
        xs.append(lm.x * width)
        ys.append(lm.y * height)

    # projection.py を使用してマッピング座標を計算
    pts = projection.mapping(xs, ys)

    # --- UV展開処理 ---
    b_channel = npy_img[:, :, 0]
    g_channel = npy_img[:, :, 1]
    r_channel = npy_img[:, :, 2]

   # バイリニア補間で展開
    uv_b = projection.inter_linear(b_channel, pts).reshape(256, 256)
    uv_g = projection.inter_linear(g_channel, pts).reshape(256, 256)
    uv_r = projection.inter_linear(r_channel, pts).reshape(256, 256)

    # 結合してBGR画像に戻す
    uv_img_raw = cv2.merge([uv_b, uv_g, uv_r])
    
    # ================== ★ここから修正：上下反転処理を追加 ==================
    # OpenCVのcv2.flip関数を使って画像を上下に反転させる
    # 第2引数の '0' が「上下反転」を意味します。 (1:左右反転, -1:上下左右反転)
    uv_img = cv2.flip(uv_img_raw, 0)
    # ====================================================================

    # =========================================================
    # ★修正: 背景の「自動塗りつぶし色」を取得し、完璧に黒(0.0)で上書きする
    # =========================================================
    # UVマップの左上角(0,0)は必ず背景になるため、その色のBGR値を取得
    bg_b = uv_img[0, 0, 0]
    bg_g = uv_img[0, 0, 1]
    bg_r = uv_img[0, 0, 2]

    # その背景色と「完全に同じ色」の領域を特定する（計算誤差を考慮して差が1e-3未満の場所）
    mask_bg = (np.abs(uv_img[:, :, 0] - bg_b) < 1e-3) & \
              (np.abs(uv_img[:, :, 1] - bg_g) < 1e-3) & \
              (np.abs(uv_img[:, :, 2] - bg_r) < 1e-3)

    # 背景部分を強制的に 0.0（真っ黒）にする
    uv_img[mask_bg] = 0.0
    # =========================================================
    
    return uv_img

def process_directory(input_dir_str, output_base_dir_str, detector):
    """
    指定ディレクトリ内の .npy ファイルをすべて処理して保存します。
    
    変更点:
    output_base_dir_str (例: .../bare) の下に、
    [元のディレクトリ名]_uvs というディレクトリを作成し、
    その中にファイルを保存します。
    """
    input_dir = Path(input_dir_str)
    output_base_dir = Path(output_base_dir_str)
    
    if not input_dir.exists():
        print(f"エラー: 入力ディレクトリが見つかりません: {input_dir}")
        return

    # --- 変更箇所開始 ---
    # 元のディレクトリ名を取得して _uvs を付与
    specific_dir_name = input_dir.name + "_uvs"
    
    # 最終的な出力先ディレクトリパスを作成
    # 例: output_uvs_texture/bare/exp001_..._bare0_p75_uvs
    final_output_dir = output_base_dir / specific_dir_name
    
    # ディレクトリ作成
    final_output_dir.mkdir(parents=True, exist_ok=True)
    # --- 変更箇所終了 ---

    npy_files = sorted(input_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"警告: .npyファイルが見つかりません: {input_dir}")
        return

    print(f"処理開始: {input_dir.name} ({len(npy_files)} files) -> {final_output_dir}")
    
    success_count = 0
    
    for npy_path in tqdm(npy_files):
        try:
            # .npy 読み込み
            frame = np.load(npy_path)
            
            # UV展開 (結果は float)
            uv_img = generate_uv_from_npy(detector, frame)
            
            # 保存ファイル名: [元のファイル名(拡張子なし)] + "_uvs.npy"
            save_name = f"{npy_path.stem}_uvs.npy"
            save_path = final_output_dir / save_name  # 変更したディレクトリに保存

            if uv_img is not None:
                # numpy形式で保存
                np.save(str(save_path), uv_img)
                success_count += 1
            else:
                # 顔が見つからない場合: 黒画像(float)を保存
                black_img = np.zeros((256, 256, 3), dtype=np.float64)
                np.save(str(save_path), black_img)
                
        except Exception as e:
            print(f"エラー発生 ({npy_path.name}): {e}")

    print(f"完了: {success_count} / {len(npy_files)} 枚を正常に変換しました。\n")

def main():
    # 1. 検出器の準備
    try:
        detector = initialize_face_landmarker()
    except Exception as e:
        print(f"初期化エラー: {e}")
        return

    # 2. 素肌データの処理
    # 出力ベース: output_uvs_texture/bare
    # 実際の保存先: output_uvs_texture/bare/[元のDir名]_uvs/
    process_directory(
        BARE_FACE_DIR, 
        Path(OUTPUT_ROOT) / "bare", 
        detector
    )

    # 3. 化粧データの処理
    # 出力ベース: output_uvs_texture/fund
    # 実際の保存先: output_uvs_texture/fund/[元のDir名]_uvs/
    process_directory(
        MAKEUP_DIR, 
        Path(OUTPUT_ROOT) / "fund", 
        detector
    )

if __name__ == "__main__":
    main()