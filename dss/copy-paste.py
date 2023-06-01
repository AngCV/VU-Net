from labelme import utils
import json
import os
from PIL import Image
import numpy as np
import cv2
from SW_datasets_utils import SW
import shutil

def edge_blur(png:np.ndarray):

    png = (png>0)*1

    if png.sum() < 100:
        blur = [0.5 , 0.8]
    elif  png.sum() < 300:
        blur = [0.5 , 0.7 , 0.9]
    else:
        blur = [0.5 , 0.6 , 0.7 , 0.9]
    mask = png
    for b in blur:
        png_temp = png * 255
        png_contours = np.zeros(png.shape,dtype=np.uint8)
        contours , _ = cv2.findContours(png_temp.astype(np.uint8) , cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(png_contours,contours,contourIdx=-1,color=1,thickness=1)
        png_contours = png_contours.astype(np.float32) * b
        mask = np.where(png_contours,png_contours,mask)
        png = png - (png_contours>0)
    return mask

def building_count(json_file):

    for i in range(len(json_file['shapes']))[::-1]:
        if not json_file['shapes'][i]['label'] == "building":
            json_file['shapes'].pop(i)

    img_shape = (json_file['imageHeight'],json_file['imageWidth'],3)
    png ,_= utils.labelme_shapes_to_label(img_shape, json_file['shapes'])
    num_of_allpixel = json_file['imageHeight'] * json_file['imageWidth']
    p100 = []
    p1_300 = []
    p300 = []

    for i in range(len(json_file['shapes'])):
        temp = []
        temp.append(json_file['shapes'][i])
        png_single, _ = utils.labelme_shapes_to_label(img_shape, temp)
        if png_single.sum() < 100:
            p100.append(png_single.sum())
        elif png_single.sum() < 300:
            p1_300.append(png_single.sum())
        else:
            p300.append(png_single.sum())



    return p100,p1_300,p300,num_of_allpixel

def random_RF(image:np.ndarray,png:np.ndarray):
    """
    对目标图片实施随机翻转旋转
    输入：
        image:np.ndarray格式，待处理的图片
        png:np.ndarray格式，待处理的图片的mask
    输出：
        image_output:np.ndarray格式，处理好的图片
        png_output:np.ndarray格式，处理好的图片的mask
    """
    option = np.random.choice(np.arange(0, 5), size=np.random.randint(0, 4), replace=False)
    for i in option:
        if i == 0:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            png = cv2.rotate(png, cv2.ROTATE_90_CLOCKWISE)
        elif i == 1:
            image = cv2.rotate(image, cv2.ROTATE_180)
            png = cv2.rotate(png, cv2.ROTATE_180)
        elif i == 2:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            png = cv2.rotate(png, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif i == 3:
            image = cv2.flip(image, 1)
            png = cv2.flip(png, 1)
        elif i == 4:
            image = cv2.flip(image, 0)
            png = cv2.flip(png, 0)
    return image , png

def img_save(img:np.ndarray,save_path:str,name:str):
    colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
              [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
              [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
              [128, 64, 12]]
    if len(img.shape) == 3:
        img = Image.fromarray(img.astype(np.uint8), 'RGB')
        savepath = save_path + '/' + name +'.tif'
        os.makedirs(save_path, exist_ok=True)
        img.save(savepath)
    elif len(img.shape) == 2:
        img = Image.fromarray(img.astype(np.uint8), 'P')
        img.putpalette(sum(colors, []))
        savepath = save_path + '/' + name + '.png'
        os.makedirs(save_path, exist_ok=True)
        img.save(savepath)

def copy_paste(img:np.ndarray, json_file, river_png:np.ndarray, per:float = 10, if_edge_blur:bool = True, if_show:bool = False):

    np.random.seed(1)
    if not isinstance(img,np.ndarray):
        img = np.asarray(img, dtype=np.uint8)
    if not isinstance(river_png, np.ndarray):
        river_png = np.asarray(river_png, dtype=np.uint8)

    for i in range(len(json_file['shapes']))[::-1]:
        if not json_file['shapes'][i]['label'] == "building":
            json_file['shapes'].pop(i)
    png ,_= utils.labelme_shapes_to_label(img.shape, json_file['shapes'])
    h, w = png.shape
    img_paste = np.zeros(img.shape, dtype=np.uint8)
    png_paste = np.zeros(png.shape, dtype=np.uint8)

    count = np.zeros(png.shape)
    river_png = (river_png==1).astype(np.uint8)

    while count.sum() / (h * w) <= (per * 0.01):
        t = count.sum() / (h * w)
        k = np.random.randint(0,len(json_file['shapes']))
        temp = []
        temp.append(json_file['shapes'][k])
        png_single, _ = utils.labelme_shapes_to_label(img.shape, temp)
        if png_single.sum() > 5000:
            if len(json_file['shapes']) <= 1:
                break
            continue

        mask_3d = np.expand_dims(png_single,2)
        mask_3d = np.repeat(mask_3d,3,2)
        img_single = img * mask_3d

        points = np.asarray(temp[0]['points'])
        x1 = int(np.floor(points[::,0].min()))
        x2 = int(np.ceil(points[::,0].max()))
        y1 = int(np.floor(points[::,1].min()))
        y2 = int(np.ceil(points[::,1].max()))

        png_single_crop = png_single[y1:y2,x1:x2].astype(np.uint8)
        img_single_crop = img_single[y1:y2,x1:x2].astype(np.uint8)

        img_single_crop,png_single_crop = random_RF(img_single_crop,png_single_crop)
        y_h, x_w = png_single_crop.shape

        mask = edge_blur(png_single_crop)
        if if_edge_blur == False:
            mask = (mask>0)*1
        mask = np.expand_dims(mask, 2)
        mask = np.repeat(mask, 3, 2)

        y_p = np.random.randint(0, h - y_h)
        x_p = np.random.randint(0, w - x_w)


        if ((png + png_paste + (river_png))[y_p : (y_p + y_h ) , x_p : (x_p + x_w )] * png_single_crop).sum() == 0:

            img_paste[y_p:(y_p+y_h),x_p:(x_p+x_w)] = img_single_crop*mask + img[y_p:(y_p+y_h),x_p:(x_p+x_w)] *(1-mask)
            png_paste[y_p:(y_p+y_h),x_p:(x_p+x_w)] = np.where(png_single_crop,png_single_crop,png_paste[y_p:(y_p+y_h),x_p:(x_p+x_w)])

        count = png + png_paste
    img_output = np.where(img_paste,img_paste,img)
    png_output = np.where(png_paste,png_paste,png)
    if if_show:
        img_output_temp = cv2.cvtColor(img_output,cv2.COLOR_RGB2BGR)
        cv2.imshow("river_png", river_png * 233)
        cv2.imshow("img_paste",img_output_temp)
        cv2.imshow("png_paste_all",(png + png_paste).astype(np.uint8) * 233)
        cv2.imshow("png_paste",png_paste * 233)
        cv2.waitKey(0)
    return img_output , png_output

def copy_paste_main(origin_path:str = "E:/" ,
                    save_path  :str = "E:/copy_paste" ,
                    if_blur:bool = True,
                    per:float = 10):

    file_list = os.listdir(origin_path + "/tif_img")
    for i in range(len(file_list))[::-1]:
        if file_list[i].split('.')[-1] == 'json':
            file_list[i] = file_list[i].split('.')[0]
        else:
            file_list.pop(i)

    print("---------------Copy-Paste Start----------------")
    for i,file_name in enumerate(file_list):
        if os.path.exists(save_path + "/{}%/tif_img/".format(per) + file_name + ".tif"):

            if i % 10 == 0:
                print("{:.2%} complete!".format(i / len(file_list)))
            continue
        img =  Image.open(origin_path + "/tif_img/{}.tif".format(file_name))
        jsons = json.load(open(origin_path + "/tif_img/{}.json".format(file_name)))
        river_png = Image.open(origin_path + "/png_mask/{}.png".format(file_name))
        img_output , png_paste = copy_paste(img,jsons,river_png,per,if_edge_blur=if_blur,if_show=False)
        img_save(img_output,save_path + "/{}%/tif_img/".format(per),file_name)
        img_save(png_paste,save_path + "/{}%/png_mask/".format(per),file_name)
        if i%10 == 0:
            print("{:.2%} complete!".format(i/len(file_list)))

    print("---------------Copy-Paste Complete----------------")

def building_num_count():

    path = ""
    file_list = os.listdir(path + "/20220409buidling/json/")
    for i in range(len(file_list))[::-1]:
        if not file_list[i].split('.')[-1] == 'json':
           file_list.pop(i)

    p100=[]
    p1_300=[]
    p300=[]
    num_of_allpixel=[]
    for f in file_list:
        file = json.load(open(path + "/20220409buidling/json/" + f))
        t_p100,t_p1_300,t_p300,t_num_of_allpixel= building_count(file)
        p100    = p100   + t_p100
        p1_300  = p1_300 + t_p1_300
        p300    = p300   + t_p300
        num_of_allpixel.append(t_num_of_allpixel)
    print(len(p100), len(p1_300), len(p300), len(num_of_allpixel))
    print(sum(p100)/sum(num_of_allpixel), sum(p1_300)/sum(num_of_allpixel), sum(p300)/sum(num_of_allpixel))

def CP_datasets_making(percent:float = 1.5,if_blur:bool = True,
                       origin_path: str = "E:/",
                       save_path: str = "E:/copy_paste/no_blur"
                       ):

    copy_paste_main(origin_path = origin_path ,
                    save_path   = save_path ,

                    if_blur = if_blur,
                    per = percent)

    sw = SW(save_path + "/{}%".format(percent))
    sw.img_slice_blend()

    sw2 = SW(save_path + "/{}%/slice_img".format(percent))
    sw2.empty_clean()
    sw2.create_trainval_list()

    from SW_datasets_utils import files_copy
    files_copy(sourse_path = "E:/",
               target_path = save_path + "/0409building&{}%copypaste_no_blur".format(percent))
    shutil.rmtree(save_path + "/0409building&{}%copypaste_no_blur/blent_img".format(percent) )
    shutil.rmtree(save_path + "/0409building&{}%copypaste_no_blur/ImageSets".format(percent))
    #这里是唯一要手动的地方，要把0409slice的数据手动复制到新要生成的文件夹中，

    sw3 = SW(save_path + "/0409building&{}%copypaste_no_blur".format(percent))
    sw3.add_copypaste_datas(save_path + "/{}%/slice_img".format(percent),prefix=percent)

if __name__=="__main__":
    CP_datasets_making(1.5,if_blur=False)







