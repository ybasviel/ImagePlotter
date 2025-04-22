from comfy_api_simplified import ComfyApiWrapper, ComfyWorkflowWrapper
import base64
import cv2
import os

def queue(image_path:str, base_url = "http://127.0.0.1:8188/"):

    with open(image_path, 'rb') as f:
        data = f.read()
    base64_img = base64.b64encode(data)

    api = ComfyApiWrapper(base_url)

    wf = ComfyWorkflowWrapper("templates/i2i_face_api.json")

    wf.set_node_param("Load Image (Base64)", "image",  base64_img.decode() )


    results = api.queue_and_wait_images(wf, "Save Image")
    for filename, image_data in results.items():
        with open("tmp/comfy.png", "wb+") as f:
            f.write(image_data)

    image = cv2.imread("tmp/comfy.png")

    os.remove("tmp/comfy.png")

    return image

if __name__ =="__main__":

    target_file="ComfyUI_00056_.png"

    queue(target_file)