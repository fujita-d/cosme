import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- 論文用グラフのフォーマット設定 ---
plt.rcParams['font.family'] = 'Arial' 
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

# ==========================================
# 1. パス設定 (※ご自身の環境に合わせて修正してください)
# ==========================================
# 従来内積 (単体)
PATH_LEGACY = Path(r"C:\Users\hil\Documents\cosme\note\legacy_inner_map.npy")
# キャリブレーション内積 (差分)
PATH_INNER  = Path(r"C:\Users\hil\Documents\cosme\note\inner_delta_map.npy")
# SNR (差分)
PATH_SNR    = Path(r"C:\Users\hil\Documents\cosme\snr_delta_map.npy")

def create_two_circle_masks(h, w, cx1, cy1, r1, cx2, cy2, r2):
    Y, X = np.ogrid[:h, :w]
    mask_roi = np.sqrt((X - cx1)**2 + (Y - cy1)**2) <= r1
    mask_bg  = np.sqrt((X - cx2)**2 + (Y - cy2)**2) <= r2
    return mask_roi, mask_bg

def calculate_robust_stats(data_map, mask_roi, mask_bg):
    data_roi = data_map[mask_roi & ~np.isnan(data_map)]
    data_bg = data_map[mask_bg & ~np.isnan(data_map)]
    
    if len(data_roi) == 0 or len(data_bg) == 0:
        return 0, 0, 0, 0, 0

    median_roi = np.median(data_roi)
    median_bg = np.median(data_bg)
    
    q75_bg, q25_bg = np.percentile(data_bg, [75, 25])
    robust_std_bg = (q75_bg - q25_bg) / 1.349
    
    # コントラストは絶対値で評価 (単体振幅と差分をフェアに比較するため)
    diff = abs(median_roi - median_bg)
    cnr = diff / (robust_std_bg + 1e-8)
    
    return median_roi, median_bg, 0, robust_std_bg, cnr

# --- ロードと実行 ---
try:
    map_legacy = np.load(PATH_LEGACY)
    map_inner = np.load(PATH_INNER)
    map_snr = np.load(PATH_SNR)
except FileNotFoundError as e:
    print(f"エラー: {e}")
    exit()

# ==========================================
# ★領域調整
# ==========================================
CX_ROI, CY_ROI, R_ROI = 170, 150, 10 # 塗布部
CX_BG, CY_BG, R_BG = 170, 126, 10    # 背景(素肌)

mask_roi, mask_bg = create_two_circle_masks(256, 256, CX_ROI, CY_ROI, R_ROI, CX_BG, CY_BG, R_BG)

# 計算実行
stat_legacy = calculate_robust_stats(map_legacy, mask_roi, mask_bg)
stat_inner = calculate_robust_stats(map_inner, mask_roi, mask_bg)
stat_snr = calculate_robust_stats(map_snr, mask_roi, mask_bg)

methods = ['Legacy Inner', 'Calibrated Inner', 'Calibrated SNR']
cnr_values = [stat_legacy[4], stat_inner[4], stat_snr[4]]

# ==========================================
# グラフ描画
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))

# 色の指定 (グレー, 青, オレンジ)
colors = ['#A0A0A0', '#4C72B0', '#DD8452']
rects = ax.bar(methods, cnr_values, width=0.6, color=colors, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Robust Contrast-to-Noise Ratio (CNR)', fontweight='bold')
ax.set_title('Evolution of Visual Clarity (Quantitative Evaluation)', fontweight='bold', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Rose Criterion (CNR=1.0) の基準線を引く
ax.axhline(y=1.0, color='red', linestyle=':', linewidth=2, label='Visibility Threshold (CNR=1.0)')
ax.legend(loc='upper left')

# バーの上に数値を表示
for rect in rects:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=14)

ax.set_ylim(0, max(cnr_values) * 1.3 if max(cnr_values) > 0 else 1)
plt.tight_layout()
plt.show()