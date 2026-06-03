from __future__ import annotations

from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
            square_2 = get_square_region(pt_2)
            # cropped_img[square_2[0][1]:square_2[1][1], square_2[0][0]:square_2[1][0]] = (0, 255, 0)

    face_mesh.close() # インスタンスを終了させる
    return  pt_2

def draw_chosen_pt(_img:cv2.Mat, landmarks, id:int): # 薬指の付け根、目の下の座標取得
    # hand 13
    # face 101

    img_h, img_w, _ = _img.shape
    lm_id = landmarks.landmark[id]
    lm_id_pos = (int(lm_id.x * img_w), int(lm_id.y * img_h))
            
    cv2.putText(_img, f"id: ({lm_id_pos[0]}, {lm_id_pos[1]})", 
                        (lm_id_pos[0] + 10, lm_id_pos[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
    
    return lm_id_pos

def get_square_region(pt:tuple): # 座標から範囲指定
    center_x, center_y = pt
    side_length = 20
    top_left = (center_x - side_length // 2, center_y - side_length // 2)
    bottom_right = (center_x + side_length // 2, center_y + side_length // 2)

    return top_left, bottom_right

# detect.py 内
def get_sample_pos_signal(img_dir: Path, start_f: int, end_f: int, fromface: bool = True): 
    print(str(img_dir))
    img_paths = sorted(list(img_dir.glob("*.npy")))
    img_paths = [str(p) for p in img_paths]

    sample_signal = []
    square = None  

    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths[:end_f])):
        img = np.load(img_path)

        if i == 0:
            # ==========================================
            # 【切り替え】領域指定モード（手動 or 自動）
            # ==========================================

            # --- パターンA: OpenCVによる手動ドラッグ決定（推奨） ---
            # ウィンドウが開き、マウスドラッグで四角形を書き、SPACEかENTERで決定します
            roi = cv2.selectROI("Select ROI (Drag -> Space/Enter)", img, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select ROI (Drag -> Space/Enter)")
            
            # roiは (x, y, w, h) で返ってくるため、今までの square ((x1,y1), (x2,y2)) の形に変換
            if roi[2] == 0 or roi[3] == 0:
                raise ValueError("領域が正しく選択されませんでした。")
            square = ((int(roi[0]), int(roi[1])), (int(roi[0] + roi[2]), int(roi[1] + roi[3])))
            # ------------------------------------------

            # --- パターンB: トラッキングによる自動検出 ---
            # if fromface:
            #     pt = detect_face(img)
            # else:
            #     pt = detect_hand(img)
            # if pt is None:
            #     raise ValueError("顔または手の検出に失敗しました。")
            # square = get_square_region(pt)
            # ------------------------------------------

            face = img[square[0][1]:square[1][1], square[0][0]:square[1][0], 1]
            accum_face = face[:, :].copy()
            face_low_cut = face[:, :].copy()

        else:
            face = img[square[0][1]:square[1][1], square[0][0]:square[1][0], 1]
            accum_face = accum_face * 0.95 + face * 0.05
            face_low_cut = face - accum_face

        if start_f <= i + 1 <= end_f + 1:
            sample_signal.append(np.mean(face_low_cut))
        elif i + 1 > end_f + 1:
            break

    # ★変更点: 信号だけでなく、決定した square も一緒に返す
    return np.array(sample_signal), square


# def get_sample_pos_signal(img_dir: Path, start_f: int, end_f: int): #毎フレーム検出
#     print(str(img_dir))
#     img_paths = sorted(img_dir.glob("*.npy"))
#     img_paths = [str(p) for p in img_paths]

#     sample_signal = []
#     accum_face = None 

#     for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
#         if i > end_f:
#             break

#         img = np.load(img_path)

#         # 毎フレーム手を検出
#         pt = detect_hand(img)
#         if pt is None:
#             print(f"手の検出に失敗しました: frame {i}")
#             continue

#         square = get_square_region(pt)
#         x1, y1 = square[0]
#         x2, y2 = square[1]

#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

#         region = img[y1:y2, x1:x2, 1]

#         if region.size == 0:
#             print(f"空の領域が検出されました: frame {i}")
#             continue

#         if accum_face is None:
#             accum_face = region.copy()
#             face_low_cut = region.copy()
#         else:
#             accum_face = accum_face * 0.95 + region * 0.05
#             face_low_cut = region - accum_face

#         if start_f <= i+1 <= end_f+1:
#             sample_signal.append(np.mean(face_low_cut))

#     return np.array(sample_signal)



# def visualize_inner(img_dir: Path, base_phase:np.ndarray, start_f:int, end_f:int, fromface:bool, save_flag:bool=False): 
#     img_paths = img_dir.glob("*.npy")
#     img_paths = [str(p) for p in img_paths]

#     raw_green_signal = []
#     filtered_green_signal = []

#     # for i, img_paths in tqdm(enumerate(sorted(img_paths)), total=len(img_paths[:end_f])): 
#     for i, img_paths in tqdm(enumerate(sorted(img_paths)), total=len(img_paths[:end_f])): 
#         img = np.load(img_paths)
#         if i == 0:
#             first_img = img[:,:,:]
#             accum_face = img[..., 1]
#             face_low_cut = img[..., 1]
#             inner = np.zeros_like(accum_face, dtype=np.float64)
#             # --- 初回フレームのデータをリストに追加 ---
#             if start_f <= i+1 <= end_f+1:
#                 # 'face' はこのスコープでは未定義なので、初回は'accum_face'の平均値を使います
#                 raw_green_signal.append(np.mean(accum_face)) 
#                 filtered_green_signal.append(np.mean(face_low_cut))
#             # --- ここまで ---
#         else:
#             face = img[..., 1]
#             accum_face = accum_face * 0.95 + face * 0.05
#             face_low_cut = face - accum_face

#         if start_f <= i+1 <= end_f+1:
#             inner += face_low_cut * base_phase[i+1-start_f]
#             # --- プロット用のデータをリストに追加 ---
#             raw_green_signal.append(np.mean(face))
#             filtered_green_signal.append(np.mean(face_low_cut))
#             # --- ここまで ---
#             # print(len(inner))
#         elif i+1 > end_f+1:
#             break


#     # # --- ここからが追加したプロット用のコードです ---
#     # # 図1: 生の緑色成分の変動をプロット
#     # fig1, ax1 = plt.subplots()
#     # ax1.plot(raw_green_signal, color='green')
#     # ax1.set_title("Raw Green Component Fluctuation")
#     # ax1.set_xlabel("Frame")
#     # ax1.set_ylabel("Amplitude")
#     # ax1.set_yticks([])  # 縦軸の目盛りを非表示
#     # plt.show()

#     # # 図2: フィルター後のface_low_cutの変動をプロット
#     # fig2, ax2 = plt.subplots()
#     # ax2.plot(filtered_green_signal, color='green')
#     # ax2.set_title("Filtered Green Component Fluctuation (face_low_cut)")
#     # ax2.set_xlabel("Frame")
#     # ax2.set_ylabel("Amplitude")
#     # ax2.set_yticks([])  # 縦軸の目盛りを非表示
#     # plt.show()
#     # # --- ここまでが追加したコードです ---

# # 生とabsの2種類
#     if fromface: #顔からリファレンスベクトルを求める場合
#             pt = detect_face(img)   
#     else:
#             pt = detect_hand(img)
#     square = get_square_region(pt)

#     # eye_coords = get_eye_landmarks(first_img)
#     # img_with_lines = first_img.copy()
#     # line_thickness = 40
#     # if eye_coords:
#     #     line_color = (0, 0, 0)
#     #     cv2.line(img_with_lines, eye_coords["left_outer"], eye_coords["right_outer"], line_color, line_thickness)

#     fig = plt.figure(figsize=(10, 5), dpi=100)
#     # ax1 = fig.add_subplot(121)
#     # ax1.imshow(cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB))
#     # ax1.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))

#     img_height, img_width = first_img.shape[:2]
#     cut_margin1_x = int(img_width * 0)#0.24
#     # cut_margin1_x2 = int(img_width * 0.08)
#     cut_margin1_y = int(img_height * 0)#0.18
#     # ax1.set_xlim(cut_margin1_x, img_width - cut_margin1_x2)
#     # ax1.set_xlim(cut_margin1_x, img_width - cut_margin1_x)
#     # ax1.set_ylim(img_height - cut_margin1_y, 0)

#     rect = plt.Rectangle(
#         (square[0][0], square[0][1]),  
#         square[1][0] - square[0][0],  
#         square[1][1] - square[0][1],  
#         linewidth=1, edgecolor='r', facecolor='none'
#     )
#     # ax1.add_patch(rect)

#     ax2 = fig.add_subplot(122)
#     gamma = 0.25
#     # ax2.imshow(np.clip(inner, 0, 255)**gamma, vmin=0, vmax=64**gamma)
#     ax2.imshow((np.clip(inner, 0, 255)/255.0)**gamma, vmax=None, vmin=0)
#     # ax2.imshow((np.clip(inner, 0, 255)/255.0)**gamma)
#     # ax2.imshow((np.clip(inner, -150, 255)/255.0))

#     inner_height, inner_width = inner.shape[:2]
#     cut_margin2_x = int(inner_width * 0)#0.24
#     # cut_margin2_x2 = int(inner_width * 0.08)
#     cut_margin2_y = int(inner_height * 0)#0.18

#     # ax2.set_xlim(cut_margin2_x, inner_width - cut_margin2_x2)
#     ax2.set_xlim(cut_margin2_x, inner_width - cut_margin2_x)
#     ax2.set_ylim(inner_height - cut_margin2_y, 0)

#     # ax1_height = first_img.shape[0]  # 画像の高さ
#     # ax1_width = first_img.shape[1]   # 画像の幅
#     # ax1.plot([ax1_width // 3, ax1_width // 3 * 2], [ax1_height // 2 - 30, ax1_height // 2 - 30], color='black', lw=10)

#     # ax2_height = inner.shape[0]  # 画像の高さ
#     # ax2_width = inner.shape[1]   # 画像の幅
#     # ax2.plot([ax2_width // 3, ax2_width // 3 * 2], [ax2_height // 2 - 30, ax2_height // 2 - 30], color='black', lw=10)
    
#     # if eye_coords:
#     #     ax2.plot(
#     #         [eye_coords["left_outer"][0], eye_coords["right_outer"][0]],
#     #         [eye_coords["left_outer"][1], eye_coords["right_outer"][1]],
#     #         color='black',
#     #         linewidth=12
#     #     )

#     # ax1.axis("off")
#     ax2.axis("off")
#     plt.subplots_adjust(wspace=0)
    
#     # output_dir = img_dir.parent / "output"
#     # output_dir.mkdir(exist_ok=True, parents=True)
#     # output_name = img_dir.name
#     # fig.savefig(
#     #     f"{output_dir}/{output_name}_abs.pdf", bbox_inches="tight", pad_inches=0.0
#     # )
#     # plt.close()
#     # return
#     # print(sample_signal[0].shape)
#     # sample_signal = np.array(sample_signal).T
#     # # plt.plot(base_signal[0], label="blue", color="blue")
#     # plt.plot(sample_signal[1], label="green", color="green")
#     # # plt.plot(base_signal[2], label="red", color="red")
#     # plt.show()

#     # plt.savefig("fig_test.png")
#     if save_flag == True:
#         output_dir = img_dir.parent / "output_"
#         output_dir.mkdir(exist_ok=True, parents=True)
#         output_name = img_dir.name
#         # output_path = f"{output_dir}/{output_name}._abs.pdf"
#         # print(f"Saving figure to: {output_path}")
#         fig.savefig(
#             f"{output_dir}/{output_name}_abs.pdf", bbox_inches="tight", pad_inches=0.0
#         )
#         plt.close()
#     else:
#         plt.show()
#     return inner
#     ...

# ★追加: カラーバー用の枠を調整するライブラリ
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ★追加: 画像グリッドを整えるためのライブラリ
from mpl_toolkits.axes_grid1 import ImageGrid

from mpl_toolkits.axes_grid1 import ImageGrid

from mpl_toolkits.axes_grid1 import ImageGrid

# 引数に square: tuple を追加します
def visualize_inner(img_dir: Path, base_phase: np.ndarray, square: tuple, start_f: int, end_f: int, fromface: bool, save_flag: bool = False): 
    img_paths = sorted(list(img_dir.glob("*.npy")))
    img_paths = [str(p) for p in img_paths]

    if not img_paths:
        print("エラー: 画像ファイル(.npy)が見つかりません。")
        return None

    first_img = None 
    inner = None

    # --- メインループ ---
    for i, path in tqdm(enumerate(img_paths), total=len(img_paths[:end_f])): 
        img = np.load(path)
        
        if i == 0:
            first_img = img.copy()
            accum_face = img[..., 1]
            face_low_cut = img[..., 1]
            inner = np.zeros_like(accum_face, dtype=np.float64)
        else:
            face = img[..., 1]
            accum_face = accum_face * 0.95 + face * 0.05
            face_low_cut = face - accum_face

        if start_f <= i+1 <= end_f+1:
            inner += face_low_cut * base_phase[i+1-start_f]
        elif i+1 > end_f+1:
            break

    if first_img is None:
        return None

    # --- 描画処理 ---
    # ★変更点: ここにあった detect_face や detect_hand を削除し、引数の square をそのまま使う

    fig = plt.figure(figsize=(14, 6))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.3,
                     cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1)

    # --- ① 左側：元のRGB画像と赤枠 ---
    grid[0].imshow(cv2.cvtColor(first_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    # 引数で受け取った square を元に赤枠を描画
    rect = plt.Rectangle(
        (square[0][0], square[0][1]),  
        square[1][0] - square[0][0],  
        square[1][1] - square[0][1],  
        linewidth=2, edgecolor='r', facecolor='none'
    )
    grid[0].add_patch(rect)
    grid[0].set_axis_off()

    # # --- ② 右側：可視化画像 ---
    # gamma = 0.25
    # disp_data = (np.clip(inner, 0, 255)/255.0)**gamma
    # im = grid[1].imshow(disp_data, vmax=0.5, vmin=0)
    # grid[1].set_axis_off()
    # --- ② 右側：可視化画像 ---
    # 1. 顔の概算領域をマウスドラッグで手動取得
    print("顔の領域（明るさ調整用）をドラッグで選択し、SpaceかEnterを押してください。")
    
    # first_img (1フレーム目の画像) を使ってROI選択ウィンドウを表示
    roi_vmax = cv2.selectROI("Select Face Region for Contrast", first_img.astype(np.uint8), showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Face Region for Contrast")
    
    x, y, w, h = roi_vmax
    
    # 万が一、間違えて選択せずにウィンドウを閉じてしまった場合のフェイルセーフ（エラー回避）
    if w == 0 or h == 0:
        print("領域が選択されなかったため、デフォルト（中央部分）を使用します。")
        h_img, w_img = inner.shape
        face_region = inner[h_img//4 : 3*h_img//4, w_img//4 : 3*w_img//4]
    else:
        # 選択された領域のデータを切り出す
        face_region = inner[y : y+h, x : x+w]
    
    # 2. 選択した顔領域の中の「実質的な最大値（99パーセンタイル）」を計算
    face_vmax = np.percentile(face_region, 99)
    
    # エラー防止（万が一ゼロ以下になった場合）
    if face_vmax <= 0:
        face_vmax = 1.0 
        
    # 3. データを 0 〜 face_vmax の範囲に収め、最大値で割って 0.0〜1.0 に正規化
    norm_inner = np.clip(inner, 0, face_vmax) / face_vmax
    
    # ガンマ補正（0.5〜0.8あたり）
    gamma = 0.5 
    disp_data = norm_inner ** gamma
    
    # 4. 表示 (vmin=0.0, vmax=1.0 に固定された状態で、綺麗なコントラストになる)
    im = grid[1].imshow(disp_data, cmap='viridis', vmin=0.0, vmax=1.0)
    
    # 軸を消す
    grid[1].set_axis_off()

    # --- ③ カラーバー ---
    grid.cbar_axes[0].colorbar(im)

    if save_flag:
        output_dir = img_dir.parent / "output_"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_name = img_dir.name
        save_path = output_dir / f"{output_name}_abs.pdf"
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()
    else:
        plt.show()
    
    return inner


if __name__ == "__main__":
    output_dir.mkdir(exist_ok=True)
    proc(exp_path=img_dir, full_size=True)