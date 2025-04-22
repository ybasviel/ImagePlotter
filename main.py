import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import time
from edges_to_vec import edges2polylines
from pathlib import Path
import tqdm

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
    edges = cv2.Canny(gray, 70, 100)

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

def dump_to_gcode_str(polylines):
    moving_z = 50
    writing_z = 40
    
    with open("templates/start.gcode", "r") as f:
        start_gcode = f.read()

    with open("templates/end.gcode", "r") as f:
        end_gcode = f.read()

    gcode = ""
    gcode += start_gcode
    gcode += f"E0;\n"

    gcode += f"G1 Z{moving_z} F240\n"

    for polyline in polylines:
        if np.shape(polyline)[0] > polyline_noise_threshold:
            gcode += "G1 F7200\n"
            gcode += f"G1 X{polyline[0][0]} Y{polyline[0][1]} Z{moving_z};\n"
            gcode += f"G1 Z{writing_z} F2400;\n"
            for coord in polyline:    
                gcode += f"G1 X{coord[0]} Y{coord[1]};\n"
            gcode += f"G1 X{polyline[-1][0]} Y{polyline[-1][1]} Z{moving_z};\n"

    gcode += f"G1 Z{moving_z} F240;\n"

    gcode += end_gcode

    return gcode

def dump_to_gcode(filename:Path|str, polylines):

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

if __name__ == "__main__":
    import argparse
    from urllib.request import urlopen
    import serial

    parser = argparse.ArgumentParser(description="pen plotter")
    parser.add_argument("-i", "--input", dest="input_file_path", type=str, help="input img file path", default="")
    parser.add_argument("-u", "--input-url", dest="url", type=str, help="input img file url", default="")
    parser.add_argument("-o", "--output-gcode-path", dest="gcode_path", type=Path, help="gcode save path", default="output.gcode")
    parser.add_argument("-s", "--serial-port", dest="serial_port", type=str, help="gcode send serial port", default="")
    args = parser.parse_args()

    polyline_noise_threshold = 10

    if args.input_file_path != "":
        image = cv2.imread(args.input_file_path)

    elif args.url != "":
        image_url = args.url
        resp = urlopen(image_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        print("spaceで撮影")
        #なにも指定がないときはカメラモード
        image = capture_image()

    image = crop_image_to_sq(image)
    image = resize_image_to_640(image)

    cv2.imshow("original", image)

    edges = get_edges(image)

    print("cでキャンセル")

    cv2.imshow("edge", edges)
    key = cv2.waitKey(0) & 0xFF

    if key == ord("c"):
        exit()
    cv2.destroyAllWindows()

    print("convert edges to vec")
    polylines = edges2polylines(edges, 10, None)
    print("done")

    polylines = map_bedsize(polylines)

    polylines = sort_matrices_by_n(polylines)

    # plot polylines
    for polyline in polylines:
        if np.shape(polyline)[0] > polyline_noise_threshold:
            plt.plot(polyline[:,0], polyline[:,1])
    plt.show()

    if args.serial_port == "":
        dump_to_gcode(args.gcode_path, polylines)
    else:
        ser = serial.Serial(args.serial_port, 115200)

        gcode = dump_to_gcode_str(polylines)
        send_gcode_by_serial(ser, gcode)



        