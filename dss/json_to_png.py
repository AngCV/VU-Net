from labelme import utils
import json
import os
from PIL import Image
import numpy as np

def json_to_png():

    json_path = "C:/Users/hasee/Desktop/test/pytorch/deeplabv3-plus-pytorch-main/datas/json"
    png_path = "C:/Users/hasee/Desktop/test/pytorch/deeplabv3-plus-pytorch-main/datas/png_mask"
    os.makedirs(png_path, exist_ok=True)
    files = os.listdir(json_path)
    colors = [[0, 0, 0], [255, 255, 255], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
              [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
              [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
              [128, 64, 12]]

    for file in files:
        file_name, extension = file.split('.')
        if extension == "json":
            json_file = json.load(open(os.path.join(json_path, file)))

            shapes = json_file['shapes']
            for i in range(len(shapes))[::-1]:
                if not shapes[i]['label'] == "building":
                    shapes.pop(i)

            img_shape = (json_file['imageHeight'], json_file['imageWidth'], 3)
            png, _ = utils.labelme_shapes_to_label(img_shape, json_file['shapes'])
            png = Image.fromarray(png.astype(np.uint8), 'P')
            png.putpalette(sum(colors, []))
            png.save(png_path + '/' + file_name + '.png')

def remove_img():

    json_path = "C:/Users/hasee/Desktop/test/pytorch/deeplabv3-plus-pytorch-main/datas/json"
    tif_path = "C:/Users/hasee/Desktop/test/pytorch/deeplabv3-plus-pytorch-main/datas/tif_img"
    json_list = os.listdir(json_path)
    for i , file in enumerate(json_list):
        if file.endswith('json'):
            json_list[i] = file[:-5]
    tif_list = os.listdir(tif_path)
    for i , img in enumerate(tif_list):
        if not img[:-4] in json_list:
            img_path = tif_path + '/' + img
            os.remove(img_path)

if __name__ == '__main__':
    json_to_png()

