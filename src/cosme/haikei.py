import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 調べたいディレクトリを設定
DIR_BEFORE = Path(r"C:\Users\hil\Documents\cosme\output_uvs_texture\bare\exp001_20260323_14_35_bare0_p70_uvs")
DIR_AFTER = Path(r"C:\Users\hil\Documents\cosme\output_uvs_texture\fund\exp001_20260323_15_16_fund3_1_p70_uvs")

def analyze_both_dirs(dir_before, dir_after):
    dirs = {"Before (素肌)": dir_before, "After (化粧)": dir_after}
    
    # 2行2列のグラフを作成
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, (label, img_dir) in enumerate(dirs.items()):
        files = sorted(list(img_dir.glob("*.npy")))
        if not files:
            print(f"[{label}] ファイルが見つかりません: {img_dir}")
            continue

        # 1フレーム目を読み込む
        img = np.load(files[0])
        g_channel = img[:, :, 1] # 脈波に使うGチャンネル
        
        max_val = np.max(g_channel)
        min_val = np.min(g_channel)
        
        # 確実に「背景」であるはずの場所（左上の角 20x20ピクセル）
        bg_corner = g_channel[0:20, 0:20]
        bg_max = np.max(bg_corner)
        bg_mean = np.mean(bg_corner)
        
        # --- コンソールへの結果出力 ---
        print(f"========== 【{label}】のデータ分析 ==========")
        print(f"■ 全体の最大値: {max_val:.2f}")
        print(f"  -> (10%閾値: {max_val * 0.1:.2f})")
        print(f"■ 背景(左上20x20)の最大値: {bg_max:.2f}")
        print(f"■ 背景(左上20x20)の平均値: {bg_mean:.2f}")
        
        if bg_max > (max_val * 0.1):
            print("  ⚠️ 警告: 背景の最大値が10%閾値を超えています！これが色がついた原因です。")
        else:
            print("  ✅ 背景は現在の10%閾値でカットできるはずです。")
        print("\n")
        
        # --- 画像の表示 (左側) ---
        ax_img = axes[i, 0]
        im = ax_img.imshow(g_channel, cmap='gray')
        ax_img.set_title(f"{label} Raw G-channel (Max: {max_val:.1f})")
        fig.colorbar(im, ax=ax_img)
        
        # --- ヒストグラムの表示 (右側) ---
        ax_hist = axes[i, 1]
        # 0より大きく、最大値の30%未満の領域（ノイズが集中する暗い部分）を抽出
        low_values = g_channel[(g_channel > 0) & (g_channel < max_val * 0.3)]
        
        ax_hist.hist(low_values, bins=50, color='blue', alpha=0.7)
        # 10%の相対閾値（赤点線）
        ax_hist.axvline(x=max_val*0.1, color='red', linestyle='--', label='10% Threshold')
        # 先ほど提案した固定の絶対閾値20（緑実線）
        ax_hist.axvline(x=20, color='green', linestyle='-', label='Absolute Threshold (20)')
        
        ax_hist.set_title(f"{label} Histogram of Low Brightness (Noise Area)")
        ax_hist.set_xlabel("Pixel Value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()

    plt.tight_layout()
    plt.show()

# 実行
analyze_both_dirs(DIR_BEFORE, DIR_AFTER)