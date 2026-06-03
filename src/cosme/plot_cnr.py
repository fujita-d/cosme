import matplotlib.pyplot as plt

# --- 論文用グラフのフォーマット設定 ---
plt.rcParams['font.family'] = 'Arial' 
plt.rcParams['font.size'] = 12

# ==========================================
# 1. データ設定
# ==========================================
volumes = [1, 2, 3]

# 3つの手法のデータ
val_conv_ip = [0.94, 0.99, 1.41] # 従来の内積 (キャリブレーションなし)
val_calib_ip = [0.93, 1.18, 1.33] # キャリブレーション内積
# val_calib_snr = [1.03, 1.61, 1.51] # キャリブレーションSNR

# ==========================================
# 2. グラフの描画設定
# ==========================================
fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)

# 折れ線グラフのプロット（マーカーの白い縁取りなどで画像を再現）
# ① 従来の内積 (ベースラインとして控えめなグレーを使用)
ax.plot(volumes, val_conv_ip, marker='o', markersize=8, 
        markeredgecolor='white', markeredgewidth=1.5, linewidth=2.5, 
        color='#95a5a6', label='Conventional Inner Product')

# ② キャリブレーション内積 (ブルー)
ax.plot(volumes, val_calib_ip, marker='o', markersize=8, 
        markeredgecolor='white', markeredgewidth=1.5, linewidth=2.5, 
        color='#1f77b4', label=r'Inner Product ($\Delta V_{IP}$)')

# ③ キャリブレーションSNR (オレンジ) ※こちらも数式フォントで統一すると綺麗です
# ax.plot(volumes, val_calib_snr, marker='o', markersize=8, 
#         markeredgecolor='white', markeredgewidth=1.5, linewidth=2.5, 
#         color='#ff7f0e', label=r'SNR ($\Delta SNR$)')

# ==========================================
# 3. データラベル（数値）の追加と位置調整
# ==========================================
for i, v in enumerate(volumes):
    # SNRとキャリブレーション内積はマーカーの「上」に数値を表示
#     ax.annotate(f'{val_calib_snr[i]:.2f}', (v, val_calib_snr[i]), 
#                 textcoords="offset points", xytext=(0, 8), ha='center', va='bottom', fontsize=10.5)
    
    ax.annotate(f'{val_calib_ip[i]:.2f}', (v, val_calib_ip[i]), 
                textcoords="offset points", xytext=(0, -8), ha='center', va='bottom', fontsize=10.5)
    
    # 従来の内積は、他と重ならないようマーカーの「下」に数値を表示
    ax.annotate(f'{val_conv_ip[i]:.2f}', (v, val_conv_ip[i]), 
                textcoords="offset points", xytext=(0, 8), ha='center', va='top', fontsize=10.5)

# ==========================================
# 4. 装飾とレイアウト調整
# ==========================================
ax.set_title('Comparison of Robust CNR', fontsize=14)
ax.set_xlabel(r'Foundation application volume ($\mu$L)', fontsize=12)
ax.set_ylabel('Robust CNR', fontsize=12)

# X軸の目盛りを1, 2, 3のみに設定
ax.set_xticks(volumes)

# Y軸の範囲を下は0から、上はラベルが見切れないよう少し余裕を持たせる
ax.set_ylim(0, 1.9)

# Y軸のみに薄いグレーのグリッド線を引く（添付画像風）
ax.yaxis.grid(True, linestyle='-', color='#e0e0e0', alpha=0.8)
ax.xaxis.grid(False)

# 凡例を左上に枠線なしで配置
ax.legend(frameon=False, loc='upper left', fontsize=11)

plt.tight_layout()
plt.show()