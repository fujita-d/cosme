import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import gaussian_filter, median_filter 
from scipy.signal import hilbert, butter, filtfilt
from pathlib import Path
from tqdm import tqdm
import os
import copy

# --- 設定項目 ---
# 素肌 (Before) のディレクトリ
DIR_BEFORE = Path(r"C:\Users\hil\Documents\cosme\output_uvs_texture\bare\exp004_20260325_14_23_bare1_p70_uvs")

# 化粧 (After) のディレクトリ
DIR_AFTER = Path(r"C:\Users\hil\Documents\cosme\output_uvs_texture\fund\exp004_20260325_14_34_fund3_1_p70_uvs")

# 参照信号を取得するROI (Before/After共通)
ROI_X, ROI_Y, ROI_W, ROI_H = 80, 130, 15, 15

# 解析フレーム数
START_F, END_F = 0, 300
FPS = 30 # カメラのフレームレート
# ----------------
def load_video_volume(img_dir: Path, start_f, end_f):
    files = sorted(list(img_dir.glob("*.npy")))
    if len(files) < end_f:
        end_f = len(files)
        
    target_files = files[start_f:end_f]
    if not target_files:
        raise FileNotFoundError(f"画像が見つかりません: {img_dir}")

    volume = []
    print(f"Loading and Smoothing from {img_dir.name}...")
    for f in tqdm(target_files):
        img = np.load(f)
        g_channel = img[:, :, 1] # Gチャンネルを取得
        
        # ==========================================
        # ★追加: UV上での空間平滑化 (どちらかを選択)
        # ==========================================
        
        # パターンA: 3x3 メディアンフィルタ (エッジを保持しつつノイズを除去 / 推奨)
        smoothed_img = median_filter(g_channel, size=(3, 3))
        
        # パターンB: ガウシアンスムージング (全体を滑らかにする場合)
        #smoothed_img = gaussian_filter(g_channel, sigma=1.0)
        
        # ------------------------------------------
        
        volume.append(smoothed_img)
        
    return np.array(volume, dtype=np.float32)

# def load_video_volume(img_dir: Path, start_f, end_f):
#     files = sorted(list(img_dir.glob("*.npy")))
#     if len(files) < end_f:
#         end_f = len(files)
        
#     target_files = files[start_f:end_f]
#     if not target_files:
#         raise FileNotFoundError(f"画像が見つかりません: {img_dir}")

#     volume = []
#     print(f"Loading from {img_dir.name}...")
#     for f in tqdm(target_files):
#         img = np.load(f)
#         volume.append(img[:, :, 1]) # Gチャンネル
        
#     return np.array(volume, dtype=np.float32)

def bandpass_filter(data, fs=30, lowcut=0.7, highcut=2.5, order=3):
    """
    バンドパスフィルタ: 脈波成分(約42〜150bpm)のみを抽出
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # フィルタリング (axis=0 が時間軸) [cite: 637-638]
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

def extract_reference_signal(video_vol, x, y, w, h):
    roi = video_vol[:, y:y+h, x:x+w]
    
    # ★変更点: np.mean ではなく np.median を使用し、空間的な外れ値の影響を排除
    raw_signal = np.median(roi, axis=(1, 2))
    
    # バンドパスフィルタで時間方向の脈波帯域のみ抽出
    ref_signal = bandpass_filter(raw_signal, fs=FPS)
    
    # 正規化
    if np.std(ref_signal) > 0:
        ref_signal = ref_signal / np.std(ref_signal)
        
    return ref_signal

def compute_snr_map(video_vol, ref_signal, block_size=1):
    """
    ブロック単位(デフォルト4x4)で時系列信号を平均化し、モザイク状のSNRマップを計算する
    """
    T, H, W = video_vol.shape
    
    # 割り切れない場合の端数処理（通常256なので割り切れますが念のため）
    new_H = H // block_size
    new_W = W // block_size
    
    print(f"  Downsampling volume from {H}x{W} to {new_H}x{new_W} (block_size={block_size})")
    
    # 4x4ごとに空間的な平均をとって、モザイク状の新しいボリュームを作る
    # [T, new_H, block_size, new_W, block_size] の形に変形し、空間軸で平均を取る
    reshaped_vol = video_vol[:, :new_H*block_size, :new_W*block_size].reshape(
        T, new_H, block_size, new_W, block_size
    )
    # block_sizeの軸(axis=2 と axis=4)で平均化
    downsampled_vol = np.mean(reshaped_vol, axis=(2, 4))
    
    # ピクセル数を減らした状態で2次元に変形 [T, new_H * new_W]
    P = new_H * new_W
    X = downsampled_vol.reshape(T, P)
    
    # バンドパスフィルタ
    print("  Applying bandpass filter to block signals...")
    X = bandpass_filter(X, fs=FPS)
    
    r = ref_signal
    r_perp = np.imag(hilbert(r))
    
    r = (r - np.mean(r)) / (np.std(r) + 1e-8)
    r_perp = (r_perp - np.mean(r_perp)) / (np.std(r_perp) + 1e-8)
    
    R_mat = np.column_stack([r, r_perp])
    
    RtR = R_mat.T @ R_mat
    RtR_inv = np.linalg.inv(RtR)
    Pseudoinverse = RtR_inv @ R_mat.T
    W_coef = Pseudoinverse @ X
    
    X_sync = R_mat @ W_coef
    E = X - X_sync
    
    S_var = np.var(X_sync, axis=0)
    N_var = np.var(E, axis=0)
    
    eps = 1e-8
    snr = S_var / (N_var + eps)
    
    # 結果は 64x64 のモザイク解像度になる
    snr_small = snr.reshape(new_H, new_W)
    
    # 元の画像サイズ(256x256)に引き伸ばして返す（最近傍補間でカクカクのモザイク状にする）
    snr_full = cv2.resize(snr_small, (W, H), interpolation=cv2.INTER_NEAREST)
    
    return snr_full

def visualize_calibration_result(snr_before, snr_after, first_img_before, first_img_after, roi_rect):
    eps = 1e-8
    snr_db_before = 10 * np.log10(snr_before + eps)
    snr_db_after = 10 * np.log10(snr_after + eps)
    
    # --- 1. 背景マスクの作成（ノイズ耐性の強化） ---
    # 画像内の最大輝度の10%未満の場所は「顔ではない（背景のノイズ）」として切り捨てる
    th_before = np.max(first_img_before) * 0.1
    th_after = np.max(first_img_after) * 0.1
    
    if first_img_before.ndim == 3:
        mask_before = first_img_before[:, :, 1] > th_before
        mask_after = first_img_after[:, :, 1] > th_after
    else:
        mask_before = first_img_before > th_before
        mask_after = first_img_after > th_after
        
    mask = mask_before & mask_after 
    
    # マスク外（背景）をNaN（非数）にして計算結果から完全に除外する
    snr_db_before[~mask] = np.nan
    snr_db_after[~mask] = np.nan
    
    # --- 2. 差分計算 ---
    delta_map = snr_db_after - snr_db_before
    
    # --- 3. カラースケール (vmin, vmax) の最適化 ---
    valid_before = snr_db_before[mask]
    valid_after = snr_db_after[mask]
    
    if len(valid_before) > 0 and len(valid_after) > 0:
        all_valid = np.concatenate([valid_before, valid_after])
        vmin_val = np.nanpercentile(all_valid, 5)
        vmax_val = np.nanpercentile(all_valid, 95)
    else:
        vmin_val, vmax_val = -10, 10
        
    delta_vmax = max(abs(np.nanpercentile(delta_map[mask], 5)), 
                     abs(np.nanpercentile(delta_map[mask], 95)))
    delta_vmax = np.clip(delta_vmax, 1.0, 10.0) 
    
    # --- 4. 可視化処理（背景を白色に統一） ---
    fig = plt.figure(figsize=(18, 6))
    # grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.3,
    #                  cbar_location="right", cbar_mode="each", cbar_size="5%", cbar_pad=0.1)
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.3)
    
    # ★追加: viridisカラーマップをコピーし、NaN(背景)の色を白(white)に設定
    cmap_custom = copy.copy(plt.get_cmap('viridis'))
    cmap_custom.set_bad(color='white') 
    
    # im0, im1, im2 の cmap を cmap_custom に変更
    im0 = grid[0].imshow(snr_db_before, cmap=cmap_custom, vmin=vmin_val, vmax=vmax_val)
    grid[0].set_title("Before SNR (dB)")
    grid[0].axis("off")
    #grid.cbar_axes[0].colorbar(im0)
    
    rect = plt.Rectangle((roi_rect[0], roi_rect[1]), roi_rect[2], roi_rect[3], linewidth=1, edgecolor='r', facecolor='none')
    grid[0].add_patch(rect)

    im1 = grid[1].imshow(snr_db_after, cmap=cmap_custom, vmin=vmin_val, vmax=vmax_val)
    grid[1].set_title("After SNR (dB)")
    grid[1].axis("off")
    #grid.cbar_axes[1].colorbar(im1)

    div_norm = plt.Normalize(vmin=-delta_vmax, vmax=delta_vmax) 
    im2 = grid[2].imshow(delta_map, cmap=cmap_custom, norm=div_norm)
    grid[2].set_title("Difference (After - Before)\nPurple: Concealed, Yellow: Visible")
    grid[2].axis("off")
    #grid.cbar_axes[2].colorbar(im2)

    plt.show()
    
    return delta_map

def main():
    try:
        vol_before = load_video_volume(DIR_BEFORE, START_F, END_F)
        ref_before = extract_reference_signal(vol_before, ROI_X, ROI_Y, ROI_W, ROI_H)
        
        vol_after = load_video_volume(DIR_AFTER, START_F, END_F)
        ref_after = extract_reference_signal(vol_after, ROI_X, ROI_Y, ROI_W, ROI_H)
        
        print("Computing Before SNR...")
        snr_map_before = compute_snr_map(vol_before, ref_before)
        
        print("Computing After SNR...")
        snr_map_after = compute_snr_map(vol_after, ref_after)
        
        first_img_before = vol_before[0]
        first_img_after = vol_after[0]
        
        print("Visualizing results...")
        delta_map = visualize_calibration_result(
            snr_map_before, 
            snr_map_after, 
            first_img_before, 
            first_img_after,
            (ROI_X, ROI_Y, ROI_W, ROI_H)
        )
        

        # ==========================================
        # ★ここに追加: SNRの差分マップを保存します
        np.save("snr_delta_map.npy", delta_map)
        print("SNRの差分マップを 'snr_delta_map.npy' として保存しました。")
        # ==========================================
        
        print("Done.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()