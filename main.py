import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import time
from edges_to_vec import edges2polylines
from pathlib import Path
import tqdm
import mediapipe as mp
import comfyui_api
from utils import douglas_peucker, map_bedsize, sort_matrices_by_n, crop_image_to_sq, remove_background, resize_image_to_640, get_edges, capture_image, dump_to_gcode, dump_to_gcode_str, send_gcode_by_serial


if __name__ == "__main__":
    import argparse
    from urllib.request import urlopen
    import serial

    parser = argparse.ArgumentParser(description="pen plotter")
    parser.add_argument("-i", "--input", dest="input_file_path", type=str, help="input img file path", default="")
    parser.add_argument("-u", "--input-url", dest="url", type=str, help="input img file url", default="")
    parser.add_argument("-o", "--output-gcode-path", dest="gcode_path", type=Path, help="gcode save path", default="output.gcode")
    parser.add_argument("-s", "--serial-port", dest="serial_port", type=str, help="gcode send serial port", default="")
    parser.add_argument("--comfy", dest="comfy", type=str, help="comfyui url", default="")
    parser.add_argument("--workflow", dest="workflow", type=str, help="comfyui workflow path", default="templates/i2i_face_api.json")
    parser.add_argument("--camera-id", dest="camera_id", type=int, help="camera id", default=0)
    args = parser.parse_args()


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
        image = capture_image(args.camera_id)

    image = crop_image_to_sq(image)

    image = remove_background(image)


    if args.comfy != "":
        cv2.imshow("cropped original", image)
        cv2.imwrite("tmp/raw.png", image)
        image = comfyui_api.queue("tmp/raw.png", args.comfy, args.workflow)
        Path("tmp/raw.png").unlink()
    
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

    # ダウンサンプリング
    print("downsampling")
    epsilon = 0.3
    downsampled_polylines = []
    for polyline in polylines:
        downsampled_polyline = douglas_peucker(polyline, epsilon)
        downsampled_polylines.append(downsampled_polyline)

    # plot polylines
    polyline_noise_threshold = 10

    # plot downsampled polylines
    for polyline in downsampled_polylines:
        plt.plot(polyline[:,0], polyline[:,1])
    plt.show()

    polylines = downsampled_polylines

    if args.serial_port == "":
        dump_to_gcode(args.gcode_path, polylines, polyline_noise_threshold)
    else:
        ser = serial.Serial(args.serial_port, 115200)

        gcode = dump_to_gcode_str(polylines)
        send_gcode_by_serial(ser, gcode)



        