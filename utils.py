import numpy as np
import cv2
from pathlib import Path
import tqdm
import time
import mediapipe as mp


def douglas_peucker(points, epsilon):
    """
    Douglas-Peuckerアルゴリズムで点群を間引く
    
    Parameters:
        points: numpy.ndarray (Nx2) - 入力点群
        epsilon: float - 許容誤差（閾値）
    
    Returns:
        numpy.ndarray (Mx2) - 簡略化された点群
    """
    # 点群の点数が2以下なら、そのまま返す
    if len(points) <= 2:
        return points
    
    # 始点と終点を結ぶ直線からの距離を計算
    def perpendicular_distance(point, line_start, line_end):
        if np.all(line_start == line_end):
            return np.linalg.norm(point - line_start)
        
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        
        # 点からラインへの射影の長さ
        projection_len = np.dot(point_vec, line_unitvec)
        
        # 点からラインへの距離
        if projection_len < 0:
            # 始点の外側
            return np.linalg.norm(point - line_start)
        elif projection_len > line_len:
            # 終点の外側
            return np.linalg.norm(point - line_end)
        else:
            # 線分上の投影点から点までの距離
            projection = line_start + line_unitvec * projection_len
            return np.linalg.norm(point - projection)
    
    # 各点から直線への距離を計算
    dists = [perpendicular_distance(points[i], points[0], points[-1]) 
             for i in range(1, len(points)-1)]
    
    # 最大距離とそのインデックスを見つける
    if dists:
        max_dist_index = np.argmax(dists)
        max_dist = dists[max_dist_index]
        max_dist_index += 1  # 計算時に始点をスキップしたため調整
    else:
        max_dist = 0
    
    # 再帰的に処理
    if max_dist > epsilon:
        # 閾値より大きい場合は、その点で分割して再帰的に処理
        left_points = douglas_peucker(points[:max_dist_index+1], epsilon)
        right_points = douglas_peucker(points[max_dist_index:], epsilon)
        
        # 結果を結合（重複を避けるため）
        simplified_points = np.vstack((left_points[:-1], right_points))
    else:
        # 閾値以下なら始点と終点のみ保持
        simplified_points = np.vstack((points[0], points[-1]))
    
    return simplified_points



def sort_matrices_by_n(matrices_list):
    """
    2xnのNumPy行列を含むリストをnの小さい順に並べ替える。

    Parameters:
    matrices_list (list of numpy.ndarray): 2xnのNumPy行列を含むリスト

    Returns:
    list of numpy.ndarray: nの小さい順に並べ替えられた行列を持つリスト
    """
    if not all(isinstance(matrix, np.ndarray) and matrix.shape[1] == 2 for matrix in matrices_list):
        raise ValueError("All elements in the list must be 2xn numpy arrays.")
    
    # n, i.e., the number of columns, is determined by the second dimension of the shape
    sorted_list = sorted(matrices_list, key=lambda x: x.shape[0])
    return sorted_list

def capture_image(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    while True:
        # フレームを取得
        ret, origin_frame = cap.read()

        frame = crop_image_to_sq(origin_frame)
        
        if not ret:
            print("フレームを取得できませんでした")
            break
        
        # フレームをウィンドウに表示
        cv2.imshow('Video Preview', cv2.flip(frame, 1))
        
        # キー入力を待機
        key = cv2.waitKey(1) & 0xFF
        
        # スペースバーが押されたら画像を保存
        if key == 32:
            cap.release()
            cv2.destroyAllWindows()
            return origin_frame




def get_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 100)

    return edges

def resize_image_to_640(image):
    (h, w) = image.shape[:2]

    # 横幅を640に変更し、縦横比を維持するための高さを計算
    new_width = 640
    aspect_ratio = new_width / float(w)
    new_height = int(h * aspect_ratio)

    return cv2.resize(image, (new_width, new_height))

def map_bedsize(polylines):
    min_val = 0
    max_val = 640

    mapped_polylines = []
    for polyline in polylines:
        mapped_polyline = (polyline - min_val) / (max_val - min_val) * 170
        mapped_polyline[:,0] += 130/2
        mapped_polyline[:,1] += 270
        mapped_polylines.append(mapped_polyline) # たぶんmax 300mm


    return mapped_polylines

def dump_to_gcode_str(polylines, polyline_noise_threshold = 10):
    moving_z = 50
    writing_z = 47
    
    with open("templates/start.gcode", "r") as f:
        start_gcode = f.read()

    with open("templates/end.gcode", "r") as f:
        end_gcode = f.read()

    gcode = ""
    gcode += start_gcode
    gcode += f"E0;\n"

    gcode += f"G1 Z{moving_z} F2000\n"

    for polyline in polylines:
        if np.shape(polyline)[0] > polyline_noise_threshold:
            gcode += "G1 F15000\n"
            gcode += f"G1 X{polyline[0][0]} Y{polyline[0][1]} Z{moving_z};\n"
            gcode += f"G1 Z{writing_z} F10000;\n"
            for coord in polyline:    
                gcode += f"G1 X{coord[0]} Y{coord[1]};\n"
            gcode += f"G1 X{polyline[-1][0]} Y{polyline[-1][1]} Z{moving_z};\n"

    gcode += f"G1 Z{moving_z} F2000;\n"

    gcode += end_gcode

    return gcode

def dump_to_gcode(filename:Path|str, polylines, polyline_noise_threshold = 10):

    gcode = dump_to_gcode_str(polylines)
    with open(filename, "w") as f:
        f.write(gcode)


def crop_image_to_sq(image):
    # 画像のサイズを取得
    height, width = image.shape[:2]

    # 中央の正方形を切り出す範囲を計算
    x_start = (width - height) // 2
    y_start = 0
    x_end = x_start + height
    y_end = y_start + height

    # 中央の正方形を切り出す
    return image[y_start:y_end, x_start:x_end]

def send_gcode_by_serial(ser, codes):
    codes = codes.split("\n")

    ser.write(b"\r\n\r\n") # Wake up microcontroller
    time.sleep(3)
    ser.reset_input_buffer()

    tqdm4codes = tqdm.tqdm(codes, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=" codes", ncols=130)
    for code in tqdm4codes:
        tqdm4codes.set_postfix(gcode=code) # Show gcode at postfix
        if code.strip().startswith(';') or code.isspace() or len(code) <=0:
            continue
        else:
            ser.write((code+'\n').encode())
            while(1): # Wait untile the former gcode has been completed.

                if ser.readline().startswith(b'ok'):
                    break

def remove_background(image):
    # MediaPipeのSelfieSegmentationモデルを初期化
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1  # 1: 高精度モード
    )

    # 画像をRGBに変換
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # セグメンテーションを実行
    results = selfie_segmentation.process(rgb_image)
    
    # セグメンテーションマスクを取得
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
    
    # 背景を黒に設定
    bg_image = np.zeros_like(image)
    
    # マスクを適用
    result = np.where(condition, image, bg_image)
    
    # リソースを解放
    selfie_segmentation.close()
    
    return result