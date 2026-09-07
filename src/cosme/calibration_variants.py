"""
calibration.py の派生版（黄色い点々ノイズ対策）。

大本の calibration.py は変更せず、共通処理（読み込み・参照信号抽出・
バンドパス・Hilbert射影）はそこから import して再利用する。
本ファイルでは以下の3手法を「1ファイル・モード切替・横並び比較」で実装する。

  - "signal_only" : 提案① SN比の N(=残差ノイズ分散)を外し、信号パワー var(X_sync) のみで評価
  - "snr_clipped" : 提案② 従来の SNR(=S/N) のまま、ΔSNR>0 を 0 にクリップ
  - "combined"    : ①+② を併用（信号のみ比 + 正クリップ + 外れ値対策）

外れ値対策（提案③）は3手法共通で適用する:
  (1) マスク強化  : 輝度閾値 + before スコアのノイズフロア閾値
  (2) 相対フロア  : dB化の際、固定epsではなく有効画素中央値の数%を下限に
  (3) パーセンタイルクリップ : 残存外れ値の頭打ち
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import median_filter

# このファイルを直接実行(python xxx.py)した場合でも cosme パッケージを
# import できるよう、親ディレクトリ(src)を sys.path に追加する。
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# --- 大本(calibration.py)の共通処理を再利用 ---
from cosme.calibration import (
    load_video_volume,
    bandpass_filter,
    extract_reference_signal,
    FPS,
)

# ==========================================
# 設定項目（calibration.py と同様）
# ==========================================
DIR_BEFORE = Path(r"C:\Users\hil\Documents\cosme\output_uvs_texture\bare\exp004_20260325_14_23_bare1_p70_uvs")
DIR_AFTER = Path(r"C:\Users\hil\Documents\cosme\output_uvs_texture\fund\exp004_20260325_14_27_fund1_1_p70_uvs")

# 参照信号を取得するROI (Before/After共通)
ROI_X, ROI_Y, ROI_W, ROI_H = 80, 130, 15, 15

# 解析フレーム数
START_F, END_F = 0, 300

# --- 外れ値対策パラメータ（提案③）---
NOISE_FLOOR_FRAC = 0.05   # before スコア中央値のこの割合未満の画素は信頼できないとして除外/下限化
LUMINANCE_TH_FRAC = 0.1   # 最大輝度のこの割合未満は背景としてマスク
CLIP_PCTL = (1, 99)       # 差分マップのパーセンタイルクリップ範囲（外れ値ガード）
CLIP_HIGH = 11.0          # クリップ手法(3,4枚目)の上限クリップ値[dB]。Δ>CLIP_HIGH を頭打ちにする
CLIP_DISPLAY_VMAX = 20.0  # クリップ手法(3,4枚目)の表示レンジ ±CLIP_DISPLAY_VMAX[dB]（固定）。
                          # CLIP_HIGH より広くすることで、頭打ち値(11)が黄色く飽和せず中間色になる

# --- 方針2: 空間ロバスト化（弱め設定。点々が残るなら値を上げる）---
BLOCK_SIZE = 2            # スコア計算時の空間ブロック平均サイズ（点々ノイズ低減）
DELTA_SMOOTH_SIGMA = 0.0  # 差分マップへのガウシアン平滑化σ（NaN対応, 0で無効）

# --- 回帰方法の選択（射影） ---
# "ols"   : 通常の最小二乗（高速・既定。外れ値に弱い）
# "huber" : Huber-IRLS ロバスト回帰（体動等の時間外れ値に頑健。やや低速）
REGRESSION = "huber"
HUBER_DELTA = 1.345       # Huberのしきい値（標準化残差）。小さいほど外れ値に厳しい
HUBER_ITERS = 5           # IRLSの反復回数

# --- 信号モード（脈波の作り方） ---
# "green" : Gチャンネルのみ（現状）
# "pos"   : POS法(Wang et al. 2017)でRGBから照明ロバストな脈波を合成
SIGNAL_MODE = "green"

# --- 方針1: 大域オフセット除去（素肌基準で中心化）---
# True: 差分マップから「顔全体のロバスト中央値(=おおむね素肌レベル)」を引き、
#       塗布部の局所的な抑制だけが浮き上がるようにする。
CENTERING = True

# --- CNR評価用の2円ROI（snrvsnaiseki.py 準拠）---
CX_ROI, CY_ROI, R_ROI = 170, 150, 10   # 塗布部
CX_BG, CY_BG, R_BG = 170, 126, 10      # 背景(素肌)

# --- 手法定義（共通処理＋フラグ差のみ）---
# clip_positive は「表示専用」。CNR評価は常に未クリップのマップで行う（方針3）。
# use_noise=True  : 従来SNR (S/N, Nを残す)
# use_noise=False : 提案① (S のみ, Nを除去)
# clip_positive=True : 提案② (Δ>0を0クリップ。表示のみ)
METHODS = {
    "conventional_snr": dict(use_noise=True, clip_positive=False,
                             title="Conventional SNR (keep N, diverging)"),
    "proposal1_noN": dict(use_noise=False, clip_positive=False,
                          title="Proposal1: Signal-only (no N, diverging)"),
    "proposal2_clip": dict(use_noise=True, clip_positive=True,
                           title="Proposal2: SNR + clip (keep N, clip)"),
    "combined": dict(use_noise=False, clip_positive=True,
                     title="Combined: no N + clip"),
}

EPS = 1e-8


# ==========================================
# スコアマップ計算
# ==========================================
def _mad_scale(resid: np.ndarray) -> np.ndarray:
    """画素ごとのロバスト散布度（1.4826×MAD）。resid: (T, P) → (P,)"""
    med = np.median(resid, axis=0)
    return 1.4826 * np.median(np.abs(resid - med), axis=0) + EPS


def _fit_huber(X: np.ndarray, r: np.ndarray, r_perp: np.ndarray):
    """
    Huber-IRLS による頑健な2予測子回帰（全画素ベクトル化）。
    x(t) = a·r + b·r⊥ + e を、外れフレームの重みを下げて推定。
    返り値: (X_sync, E)  いずれも (T, P)
    """
    T, P = X.shape
    w = np.ones((T, P))
    Xs = None
    for _ in range(HUBER_ITERS):
        # 重み付き2x2正規方程式を画素ごとに解く
        a11 = (w * (r * r)[:, None]).sum(0)
        a12 = (w * (r * r_perp)[:, None]).sum(0)
        a22 = (w * (r_perp * r_perp)[:, None]).sum(0)
        b0 = (w * r[:, None] * X).sum(0)
        b1 = (w * r_perp[:, None] * X).sum(0)
        det = a11 * a22 - a12 * a12 + EPS
        W0 = (a22 * b0 - a12 * b1) / det
        W1 = (-a12 * b0 + a11 * b1) / det
        Xs = r[:, None] * W0 + r_perp[:, None] * W1
        resid = X - Xs
        # Huber重みの更新
        scale = _mad_scale(resid)
        z = np.abs(resid) / scale
        w = np.where(z <= HUBER_DELTA, 1.0, HUBER_DELTA / np.maximum(z, EPS))
    return Xs, (X - Xs)


def compute_score_components(video_vol: np.ndarray, ref_signal: np.ndarray,
                             block_size: int = 1, method: str = None):
    """
    Hilbert直交基底への射影を【1回だけ】行い、ブロック解像度の
    信号分散 S と ノイズ分散 N を返す（リサイズ前）。

    method="ols"   : 通常の最小二乗（既定。N=var(E)）
    method="huber" : Huber-IRLSロバスト回帰（N はロバスト散布度(MAD)で算出）
    method=None のときはモジュール設定 REGRESSION を使用。

    返り値: (S_small, N_small, (H, W))  ※ S_small/N_small は (new_H, new_W)
    これにより S/N（従来SNR）と S（提案①）を再射影なしで両方求められる。
    """
    method = (method or REGRESSION).lower()

    T, H, W = video_vol.shape

    new_H = H // block_size
    new_W = W // block_size

    reshaped = video_vol[:, :new_H * block_size, :new_W * block_size].reshape(
        T, new_H, block_size, new_W, block_size
    )
    downsampled = np.mean(reshaped, axis=(2, 4))

    P = new_H * new_W
    X = downsampled.reshape(T, P)

    # 時間方向のバンドパス
    X = bandpass_filter(X, fs=FPS)

    # 参照信号と90度位相シフト信号で直交基底を作成
    r = ref_signal
    r_perp = np.imag(hilbert(r))
    r = (r - np.mean(r)) / (np.std(r) + EPS)
    r_perp = (r_perp - np.mean(r_perp)) / (np.std(r_perp) + EPS)

    if method == "huber":
        # ロバスト回帰：時間外れ値の影響を抑制
        X_sync, E = _fit_huber(X, r, r_perp)
        S_small = np.var(X_sync, axis=0).reshape(new_H, new_W)
        # N もロバスト散布度で（外れ値での過大評価を防ぐ）
        N_small = (_mad_scale(E) ** 2).reshape(new_H, new_W)
    elif method == "ols":
        # 最小二乗射影（高速・全画素一括）
        R_mat = np.column_stack([r, r_perp])
        RtR_inv = np.linalg.inv(R_mat.T @ R_mat)
        W_coef = (RtR_inv @ R_mat.T) @ X
        X_sync = R_mat @ W_coef
        E = X - X_sync
        S_small = np.var(X_sync, axis=0).reshape(new_H, new_W)
        N_small = np.var(E, axis=0).reshape(new_H, new_W)
    else:
        raise ValueError(f"未知の回帰方法: {method!r} ('ols' または 'huber')")

    return S_small, N_small, (H, W)


def _resize_score(score_small: np.ndarray, hw) -> np.ndarray:
    """ブロック解像度のスコアを線形補間で元サイズへ（モザイクの格子を滑らかに）。"""
    H, W = hw
    return cv2.resize(score_small, (W, H), interpolation=cv2.INTER_LINEAR)


def score_from_components(S_small, N_small, hw, use_noise: bool) -> np.ndarray:
    """事前計算した S/N 成分から最終スコアマップ（リサイズ済み）を作る。

    use_noise=True  -> S/N （従来SNR）
    use_noise=False -> S   （提案①）
    ※ リサイズは「最終スコア」に対して行うため、従来 compute_score_map と結果一致。
    """
    if use_noise:
        score_small = S_small / (N_small + EPS)
    else:
        score_small = S_small
    return _resize_score(score_small, hw)


def compute_score_map(video_vol: np.ndarray, ref_signal: np.ndarray,
                      use_noise: bool = True, block_size: int = 1) -> np.ndarray:
    """後方互換ラッパー（単発呼び出し用）。内部で成分計算→スコア化。"""
    S_small, N_small, hw = compute_score_components(video_vol, ref_signal, block_size)
    return score_from_components(S_small, N_small, hw, use_noise)


# ==========================================
# POS法（照明ロバストな脈波合成）  ※calibration.py は変更しない
# ==========================================
def load_rgb_volume(img_dir: Path, start_f: int, end_f: int) -> np.ndarray:
    """
    .npy(BGR)群を読み込み、[R,G,B]順の (T,H,W,3) float32 を返す。
    load_video_volume と揃えて各チャンネルに3x3メディアンを適用。
    """
    files = sorted(list(Path(img_dir).glob("*.npy")))
    if len(files) < end_f:
        end_f = len(files)
    files = files[start_f:end_f]
    if not files:
        raise FileNotFoundError(f"画像が見つかりません: {img_dir}")

    vol = []
    print(f"Loading RGB and smoothing from {Path(img_dir).name}...")
    for f in files:
        im = np.load(str(f))
        if im.ndim != 3 or im.shape[2] < 3:
            raise ValueError(f"POSにはRGB(3ch)が必要です: {f}")
        # .npyはBGR → [R,G,B]に並べ替え + 各chに3x3メディアン
        r = median_filter(im[:, :, 2], size=(3, 3))
        g = median_filter(im[:, :, 1], size=(3, 3))
        b = median_filter(im[:, :, 0], size=(3, 3))
        vol.append(np.stack([r, g, b], axis=-1))
    return np.array(vol, dtype=np.float32)


def pos_pulse_volume(rgb_vol: np.ndarray) -> np.ndarray:
    """
    POS法(Plane-Orthogonal-to-Skin, Wang et al. 2017)で画素ごとの
    照明ロバストな脈波を合成し、(T,H,W) を返す。
    クリップ全体を1窓として時間正規化するグローバル簡易版。
    """
    T, H, W, _ = rgb_vol.shape
    C = rgb_vol.reshape(T, H * W, 3).astype(np.float64)   # (T,P,3) = [R,G,B]

    # 1. 時間正規化: 各画素・各chを時間平均で割る(照明強度スケールを除去)
    mu = C.mean(axis=0)                                   # (P,3)
    Cn = C / (mu[None, :, :] + EPS)                       # (T,P,3)

    # 2. 平面投影: P=[[0,1,-1],[-2,1,1]] を [R,G,B] に適用
    R, G, B = Cn[:, :, 0], Cn[:, :, 1], Cn[:, :, 2]
    S1 = G - B
    S2 = G + B - 2.0 * R

    # 3. チューニング: h = S1 + (std(S1)/std(S2))*S2
    std1 = S1.std(axis=0); std2 = S2.std(axis=0)
    alpha = std1 / (std2 + EPS)                           # (P,)
    h = S1 + alpha[None, :] * S2                          # (T,P)
    h = h - h.mean(axis=0, keepdims=True)                 # 平均0

    # 背景(平均輝度が極小)や非有限をゼロに
    valid = mu.mean(axis=1) > EPS
    h[:, ~valid] = 0.0
    h[~np.isfinite(h)] = 0.0
    return h.reshape(T, H, W).astype(np.float32)


def build_analysis_volume(img_dir: Path, start_f: int, end_f: int):
    """
    SIGNAL_MODE に応じて解析用の脈波ボリューム (T,H,W) と、
    マスク用の輝度先頭フレーム(2D or 3D) を返す。下流はこのボリュームを共通利用。
    """
    if SIGNAL_MODE == "pos":
        rgb = load_rgb_volume(img_dir, start_f, end_f)
        vol = pos_pulse_volume(rgb)
        lum_first = rgb[0]                # (H,W,3) 輝度はch1(G)をマスクに使用
        return vol, lum_first
    elif SIGNAL_MODE == "green":
        vol = load_video_volume(img_dir, start_f, end_f)
        return vol, vol[0]               # Gそのものが輝度
    else:
        raise ValueError(f"未知の SIGNAL_MODE: {SIGNAL_MODE!r} ('green' または 'pos')")


# ==========================================
# NaN対応のガウシアン平滑化（方針2）
# ==========================================
def nan_gaussian(arr: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """マスク外(NaN)を巻き込まずに平滑化する正規化畳み込み。"""
    if sigma <= 0:
        return arr
    from scipy.ndimage import gaussian_filter
    a = np.where(mask, np.nan_to_num(arr, nan=0.0), 0.0)
    w = mask.astype(np.float64)
    num = gaussian_filter(a, sigma)
    den = gaussian_filter(w, sigma)
    out = num / np.maximum(den, EPS)
    out[~mask] = np.nan
    return out


# ==========================================
# 差分マップ（キャリブレーション + 外れ値対策 + 中心化 + 平滑化）
# ※ クリップ(提案②)は表示専用のため、ここでは行わない（方針3）。
# ==========================================
def compute_delta(score_before: np.ndarray, score_after: np.ndarray,
                  first_img_before: np.ndarray, first_img_after: np.ndarray):
    """
    before/after のスコアマップから、マスク・フロア・dB化・差分・外れ値ガード・
    大域オフセット除去(中心化)・空間平滑化を行い、未クリップの Δ(dB) マップを返す。

    返り値: (delta_map, mask)
    """
    score_before = score_before.astype(np.float64).copy()
    score_after = score_after.astype(np.float64).copy()

    # --- (1) マスク強化: 輝度閾値 ---
    def luminance_mask(first_img):
        ch = first_img[:, :, 1] if first_img.ndim == 3 else first_img
        return ch > (np.max(first_img) * LUMINANCE_TH_FRAC)

    mask = luminance_mask(first_img_before) & luminance_mask(first_img_after)

    # --- (1') ノイズフロア（案B: マスク除外はせず、dB安定化のクランプ値としてのみ使用）---
    # 旧: mask = mask & (score_before >= floor)  でNaN穴(目・縁の白もや)が発生していた。
    # 案B では低信号画素も除外せず残す（クランプのみ）ことで白抜けを無くす。
    valid_before = score_before[mask]
    if valid_before.size > 0:
        med_before = np.median(valid_before)
        floor = max(med_before * NOISE_FLOOR_FRAC, EPS)
    else:
        floor = EPS

    # --- (2) 相対フロア + dB化 ---
    score_before = np.maximum(score_before, floor)
    score_after = np.maximum(score_after, floor)
    db_before = 10 * np.log10(score_before)
    db_after = 10 * np.log10(score_after)

    # --- 差分（After - Before）---
    delta_map = db_after - db_before
    delta_map[~mask] = np.nan

    # --- (3) パーセンタイルクリップ（残存外れ値の頭打ち。中心化前に実施）---
    valid_delta = delta_map[mask]
    if valid_delta.size > 0:
        lo, hi = np.nanpercentile(valid_delta, CLIP_PCTL)
        delta_map = np.clip(delta_map, lo, hi)
        delta_map[~mask] = np.nan

    # --- 方針1: 大域オフセット除去（顔全体のロバスト中央値≒素肌レベルを0に） ---
    if CENTERING:
        global_med = np.nanmedian(delta_map[mask])
        delta_map = delta_map - global_med
        delta_map[~mask] = np.nan

    # --- 方針2: 空間平滑化（点々ノイズ低減）---
    delta_map = nan_gaussian(delta_map, mask, DELTA_SMOOTH_SIGMA)

    return delta_map, mask


# ==========================================
# CNR計算（snrvsnaiseki.py 準拠）
# ==========================================
def _two_circle_masks(h, w):
    Y, X = np.ogrid[:h, :w]
    mask_roi = np.sqrt((X - CX_ROI) ** 2 + (Y - CY_ROI) ** 2) <= R_ROI
    mask_bg = np.sqrt((X - CX_BG) ** 2 + (Y - CY_BG) ** 2) <= R_BG
    return mask_roi, mask_bg


def robust_cnr(delta_map: np.ndarray) -> float:
    """Robust CNR = |median_roi - median_bg| / (背景のロバスト標準偏差)"""
    h, w = delta_map.shape
    mask_roi, mask_bg = _two_circle_masks(h, w)

    data_roi = delta_map[mask_roi & ~np.isnan(delta_map)]
    data_bg = delta_map[mask_bg & ~np.isnan(delta_map)]
    if data_roi.size == 0 or data_bg.size == 0:
        return 0.0

    median_roi = np.median(data_roi)
    median_bg = np.median(data_bg)
    q75, q25 = np.percentile(data_bg, [75, 25])
    robust_std_bg = (q75 - q25) / 1.349
    return abs(median_roi - median_bg) / (robust_std_bg + EPS)


# ==========================================
# 元RGB画像の表示（1窓目）
# ==========================================
def show_original_rgb(img_dir: Path, title: str = "Original RGB (after)"):
    """化粧後ディレクトリの先頭フレームを元RGB画像として表示。参照ROIを赤枠で描画。"""
    from matplotlib.patches import Rectangle
    files = sorted(Path(img_dir).glob("*.npy"))
    if not files:
        print(f"元RGB表示: 画像が見つかりません: {img_dir}")
        return
    img = np.load(str(files[START_F]))
    disp = np.clip(img, 0, 255).astype(np.uint8)
    if disp.ndim == 3 and disp.shape[2] == 3:
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)  # .npyはBGR
        # 2窓目以降と同じ輝度マスクで背景(顔以外)を白に
        g = img[:, :, 1]
    else:
        rgb = np.stack([disp] * 3, axis=-1)
        g = img
    mask = g > (np.max(img) * LUMINANCE_TH_FRAC)  # True=顔
    rgb = rgb.copy()
    rgb[~mask] = 255  # 背景を白

    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        fig.canvas.manager.set_window_title("original_rgb")
    except Exception:
        pass
    ax.imshow(rgb)
    # 参照信号ROIを赤枠で表示（論文パネル(a)相当）
    ax.add_patch(Rectangle((ROI_X, ROI_Y), ROI_W, ROI_H,
                           fill=False, edgecolor="red", linewidth=2))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    plt.show()  # 閉じると次のウィンドウ（各手法マップ）が表示される


# ==========================================
# 可視化（3手法を横並び比較）
# ==========================================
def visualize_variants(score_before_snr, score_after_snr,
                       score_before_sig, score_after_sig,
                       first_img_before, first_img_after):
    """
    3手法の Δマップを「1手法ずつ別ウィンドウ」で順番にポップアップ表示し、CNRを返す。
    ウィンドウを閉じると次の手法が表示される。
    snr系/signal系のスコアは事前に計算して渡す（重い射影計算の重複を避けるため）。
    """
    results = {}

    # viridis(紫〜緑〜黄)。背景NaNは白。
    cmap_div = copy.copy(plt.get_cmap("viridis"))   # 非クリップ：0中心の発散表示に使用
    cmap_div.set_bad(color="white")
    cmap_seq = copy.copy(plt.get_cmap("viridis"))   # クリップ：抑制量(0=紫〜強=黄)
    cmap_seq.set_bad(color="white")

    for name, cfg in METHODS.items():
        use_noise = cfg["use_noise"]
        clip_positive = cfg["clip_positive"]

        sb = score_before_snr if use_noise else score_before_sig
        sa = score_after_snr if use_noise else score_after_sig

        # --- 正準マップ（未クリップ・中心化・平滑化済み）---
        delta_map, _ = compute_delta(sb, sa, first_img_before, first_img_after)

        # --- 方針3: CNRは必ず未クリップの正準マップで評価 ---
        cnr = robust_cnr(delta_map)
        results[name] = (delta_map, cnr)

        # --- 表示用データの作成 ---
        # 表示スケールは「未クリップΔ」のロバスト95%tile|Δ|で決め、0中心の対称スケールに統一。
        # これにより 0(≒素肌) は viridis中間色(緑)、抑制(負)=紫、増加(正)=黄 となり、
        # クリップ時も背景が黄色に張り付く「白飛び」を防ぐ。
        valid = delta_map[~np.isnan(delta_map)]
        if valid.size == 0:
            disp, vmin, vmax = delta_map, -1.0, 1.0
        elif clip_positive:
            # 3,4枚目: 飛んでいる高Δ画素を CLIP_HIGH で上限クリップ(ウィンソライズ)。
            # 表示レンジは固定 ±CLIP_DISPLAY_VMAX(>CLIP_HIGH) とし、頭打ち値(11)が
            # 黄色く飽和せず中間色になるようにする。
            disp = np.clip(delta_map, None, CLIP_HIGH)
            disp[np.isnan(delta_map)] = np.nan
            vmin, vmax = -CLIP_DISPLAY_VMAX, CLIP_DISPLAY_VMAX
        else:
            # 1,2枚目: クリップなし。95%tile|Δ|の対称スケール。
            dvmax = max(float(np.nanpercentile(np.abs(valid), 95)), 1e-3)
            disp = delta_map
            vmin, vmax = -dvmax, dvmax
        cmap_use = cmap_div

        # 手法ごとに独立した図（カラーバーのかぶり防止）
        fig, ax = plt.subplots(figsize=(7, 6))
        try:
            fig.canvas.manager.set_window_title(name)
        except Exception:
            pass
        im = ax.imshow(disp, cmap=cmap_use, vmin=vmin, vmax=vmax)
        ax.set_title(f"{cfg['title']}\nRobust CNR = {cnr:.2f}", fontsize=12)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.show()  # 閉じると次の手法のウィンドウが表示される

    return results


# ==========================================
# メイン
# ==========================================
def main():
    try:
        print(f"=== Loading volumes (signal_mode={SIGNAL_MODE}) ===")
        vol_before, lum_before = build_analysis_volume(DIR_BEFORE, START_F, END_F)
        vol_after, lum_after = build_analysis_volume(DIR_AFTER, START_F, END_F)

        ref_before = extract_reference_signal(vol_before, ROI_X, ROI_Y, ROI_W, ROI_H)
        ref_after = extract_reference_signal(vol_after, ROI_X, ROI_Y, ROI_W, ROI_H)

        # 射影は before/after で各1回だけ実行し、S/N と S を使い回す（軽量化・結果不変）
        print(f"=== Projecting once per volume (block_size={BLOCK_SIZE}, regression={REGRESSION}) ===")
        Sb, Nb, hw = compute_score_components(vol_before, ref_before, block_size=BLOCK_SIZE)
        Sa, Na, _ = compute_score_components(vol_after, ref_after, block_size=BLOCK_SIZE)

        # 成分から各スコアマップを生成（再射影なし）
        snr_before = score_from_components(Sb, Nb, hw, use_noise=True)
        snr_after = score_from_components(Sa, Na, hw, use_noise=True)
        sig_before = score_from_components(Sb, Nb, hw, use_noise=False)
        sig_after = score_from_components(Sa, Na, hw, use_noise=False)

        # マスク用の輝度画像（POSでは vol[0] は脈波値なのでG輝度を使う）
        first_img_before = lum_before
        first_img_after = lum_after

        # 1窓目: 元RGB画像（化粧後・参照ROI赤枠）
        print("=== Showing original RGB (window 1) ===")
        show_original_rgb(DIR_AFTER)

        print("=== Visualizing variants ===")
        results = visualize_variants(
            snr_before, snr_after, sig_before, sig_after,
            first_img_before, first_img_after,
        )

        # CNRサマリ
        print("\n--- Robust CNR summary ---")
        for name, (_, cnr) in results.items():
            print(f"  {name:12s}: {cnr:.3f}")

        # 各手法の差分マップを保存（後段の比較用）
        for name, (delta_map, _) in results.items():
            out = f"delta_map_{name}.npy"
            np.save(out, delta_map)
            print(f"saved: {out}")

        print("Done.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
