import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import shutil
import cv2

class SW():
    def __init__(self , data_path = 'datas'):

        if os.path.isabs(data_path):
            self.data_path = data_path
        else:
            self.root_path = self.get_root_path()
            self.data_path = self.root_path + '/' + data_path

        self.tif_path = os.path.join(self.data_path , 'tif_img')
        if not os.path.exists(self.tif_path):
            os.makedirs(self.tif_path)
        self.png_path = os.path.join(self.data_path , 'png_mask')
        if not os.path.exists(self.png_path):
            os.makedirs(self.png_path)

        self.file_list = os.listdir(self.tif_path)
        for i  in range(len(self.file_list))[::-1]:
            if self.file_list[i].split('.')[-1] == 'tif' or self.file_list[i].split('.')[-1] == 'jpg':
                self.file_list[i] = self.file_list[i][:-4]
            else:
                self.file_list.pop(i)

        self.img_num = len(self.file_list)

        self.classes = ['building']
        self.colors = [[0, 0, 0],[255,255,255],[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
                        [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
                        [128, 64, 12]]

    def get_root_path(self):
        data_path = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.abspath(os.path.join(data_path , '..'))
        return data_path

    def read_img(self , imgname:str):
        try:
            img = Image.open(self.tif_path + '/' + imgname + '.tif')
        except:
            img = Image.open(self.tif_path + '/' + imgname + '.jpg')
        return img

    def read_png(self , imgname:str):
        try:
            png = Image.open(os.path.join(self.png_path, imgname + '.png'))
        except:
            png = cv2.imread(os.path.join(self.png_path, imgname + '.tif'),flags = 2)
            png = (png[:,:] > 0) * 1
        return png

    def p_n_sample_count(self):
        count = 0
        all = 0
        for file in self.file_list:
            png = self.read_png(file)
            png_np = np.asarray(png)
            count += sum(sum(png_np[:,:] == 1))
            all += png_np.shape[0] * png_np.shape[1]
        print("The Proportion of positive samples is {:.5%}".format((count / all)))

    def accuracy_count(self , ori_png:np.array , pre_png:np.array , ifprint:bool = False ):
        if isinstance(ori_png,Image.Image):
            ori_png = np.asarray(ori_png)
        if isinstance(pre_png,Image.Image):
            pre_png = np.asarray(pre_png)

        pr_png  = pre_png


        T_pixs = (pr_png == ori_png) * 1
        TP_pixs = pr_png * ori_png
        TN_pixs = T_pixs - TP_pixs
        FP_pixs = (pr_png - TP_pixs)
        FN_pixs = (ori_png - TP_pixs)
        output = TP_pixs * 1 + FP_pixs * 2 + FN_pixs * 3

        TP = np.sum(TP_pixs)
        TN = np.sum(TN_pixs)
        FP = np.sum(FP_pixs)
        FN = np.sum(FN_pixs)
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        iou = TP / (TP + FP + FN + 1e-10)
        f1_score = 2 * ( precision * recall ) / ( precision + recall + 1e-10)

        if ifprint:
            print("IoU:{:.5%}\n准确率为：{:.5%}\n精确率为：{:.5%}\n召回率为：{:.5%}".format(iou, accuracy, precision, recall))

        output = Image.fromarray(output.astype(np.uint8), mode='P')
        output.putpalette(sum(self.colors , []))

        return {"predit_image":output , "iou":iou , "accuracy":accuracy , "precision":precision ,
                "recall":recall ,"paramaters":(TP,TN,FP,FN) ,"f1_score":f1_score}

    def mask_show(self, mask:np.array , ifprint:bool = True):
        if torch.is_tensor(mask):
            mask = transforms.ToPILImage()(mask.type(torch.uint8))
        if isinstance(mask , np.ndarray):
            if len(mask.shape) == 2:
                mask = Image.fromarray(mask.astype(np.uint8) , mode= 'P')
                mask.putpalette(sum(self.colors, []))
                if ifprint:
                    mask.show()
        elif Image.isImageType(mask):
            if len(mask.size) == 2:
                mask.convert('P')
                mask.putpalette(sum(self.colors, []))
                if ifprint:
                    mask.show()
        else:
            print("img datetype error!")
            return -1

        return mask


    def img_blend_show(self,img:str , png:Image.Image , ifprint:bool = True , ifsave:bool = False):
        image = self.read_img(img)
        if isinstance(png , np.ndarray):
            png = Image.fromarray(png.astype(np.uint8),mode="P")
        if not (png.mode == 'P'):
            png.convert("P")
        png.putpalette(sum(self.colors, []))

        output = Image.blend(image, png.convert('RGB'), 0.5)
        if ifprint:
            output.show()

        #在文件名后面加上正负样本比例后缀
        png_temp = np.asarray(png)
        p_n_1 = sum(sum(png_temp[:,:] == 1))
        p_n_2 =(png.size[0] * png.size[1])
        p_n =  p_n_1 / p_n_2

        if ifsave:
            save_path = self.data_path + "/blent_img/"
            os.makedirs(save_path, exist_ok=True)
            output.save(save_path + img +"_{:.2%}.jpg".format(p_n))

        return output

    def img_pad_unless_nosize(self , img:Image.Image , size:int = 256):
        origin_w , origin_h = img.size

        if (origin_w >= size) & (origin_h >= size):
            return img ,0,0

        pad_w = 0
        pad_h = 0

        if origin_w < size :
            pad_w = size - origin_w
        if origin_h < size:
            pad_h = size - origin_h

        img_padded = Image.new(img.mode , ((origin_w + pad_w) , (origin_h + pad_h)) )
        img_padded.paste(img , (pad_w//2 , pad_h//2))

        return img_padded , pad_w , pad_h

    def img_slice_overlap(self , img:Image.Image , size:int = 256 , overlap:int = 50):

        origin_w , origin_h = img.size
        if (origin_w < size) or (origin_h < size):
            print("Error! The size of image is {} * {} ,didn't match the requirement of greater or equal to the size of {}".format(origin_w,origin_h,size))
            return -1

        w_point = []
        h_point = []
        x = 0
        y = 0
        while True:
            if x + size < origin_w:
                w_point.append(x)
                x = x + size - overlap
            else:
                x = origin_w - size
                w_point.append(x)
                break
        while True:
            if y + size < origin_h:
                h_point.append(y)
                y = y + size - overlap
            else:
                y = origin_h - size
                h_point.append(y)
                break

        imgs = []
        coordinate = []
        for j in h_point:
            for i in w_point:
                box = [i , j , i + size , j + size]
                img_temp = img.crop(box)
                imgs.append(img_temp)
                coordinate.append((i ,j))

        return imgs , coordinate

    def mask_merge_overlap(self , masks , ori_img_size , coordinates , num_classes:int = 2 , size:int = 256 , overlap:int = 50):
        predict_count = np.zeros((num_classes,ori_img_size[1],ori_img_size[0]))
        mask_weight   = np.zeros((ori_img_size[1],ori_img_size[0]),dtype=np.int32)
        for i in range(len(masks)) :
            mask = masks[i]
            y , x =  coordinates[i]
            mask_temp = np.zeros((num_classes,ori_img_size[1],ori_img_size[0]))
            xe = x + size
            ye = y + size
            t = mask_temp[: , x:xe, y:ye]
            mask_temp[: , x:xe, y:ye] = mask
            predict_count = predict_count + mask_temp

            mask_weight = mask_weight + np.round(np.sum(mask_temp,axis=0))

        assert mask_weight.any() > 0
        mask_merge = predict_count / mask_weight
        predict = mask_merge.argmax(axis=0)

        return predict

    def img_unpad_unless_nosize(self , img_padded:Image.Image , pad_w:int , pad_h:int):
        if isinstance(img_padded , np.ndarray):
            if len(img_padded.shape) ==2:
                img_padded = Image.fromarray(img_padded.astype(np.uint8),mode="P")
            else:
                img_padded = Image.fromarray(img_padded, mode="RGB")
        ori_w , ori_h = img_padded.size
        box = (pad_w//2 , pad_h//2 , ori_w - (pad_w - pad_w // 2) , ori_h - (pad_h - pad_h // 2))
        output = img_padded.crop(box)

        return output

    def img_slice_blend(self , size:int = 256 , overlap:int = 50):
        img_save_path = os.path.join(self.data_path,"slice_img/tif_img")
        os.makedirs(img_save_path, exist_ok=True)
        png_save_path = os.path.join(self.data_path,"slice_img/png_mask")
        os.makedirs(png_save_path, exist_ok=True)
        img_blend_save_path = os.path.join(self.data_path,"slice_img/blent_img")
        os.makedirs(img_blend_save_path, exist_ok=True)

        img_list = self.file_list

        print("---------Start Slicing----------")
        for k , file in enumerate(img_list):
            try:
                img = self.read_img(file)
                png = self.read_png(file)
            except:
                print("Open {} error!".format(file))
            else:
                img ,_,_= self.img_pad_unless_nosize(img , size = size)
                imgs , imgs_cor = self.img_slice_overlap(img ,size=size ,overlap=overlap)
                png_t = np.asarray(png)
                png = Image.fromarray(png_t.astype(np.uint8) , mode= 'P')
                png.putpalette(sum(self.colors, []))

                png ,_,_= self.img_pad_unless_nosize(png, size=size)
                pngs , pngs_cor = self.img_slice_overlap(png, size=size, overlap=overlap)

                for i in range(len(pngs)):

                    imgs[i].save(os.path.join(img_save_path, file + '_{}_{}.tif'.format(imgs_cor[i][0] , imgs_cor[i][1])))
                    pngs[i].save(os.path.join(png_save_path, file + '_{}_{}.png'.format(pngs_cor[i][0] , pngs_cor[i][1])))

                    img_blend = Image.blend(imgs[i] , pngs[i].convert('RGB') , 0.3)
                    img_blend.save(os.path.join(img_blend_save_path , file + '_{}_{}.tif'.format(imgs_cor[i][0] , imgs_cor[i][1])))
            if k%100 == 0:
                print("{} images have completed!{:.2%}!".format(k , float(k/len(img_list))))
        print("-----------Slicing Complete!------------")
    def img_pad(self , image, pad_size=256):
        orimg_w, orimg_h = image.size
        image_w = (orimg_w // pad_size + 1) * pad_size
        image_h = (orimg_h // pad_size + 1) * pad_size
        pad_w = image_w - orimg_w
        pad_h = image_h - orimg_h

        image_paded = Image.new(image.mode, (image_w, image_h), (0, 0, 0))
        image_paded.paste(image, (pad_w // 2, pad_h // 2))

        return image_paded, pad_w, pad_h

    def img_unpad(self , image_paded, pad_w, pad_h):
        orimg_w, orimg_h = image_paded.size

        box = (pad_w // 2, pad_h // 2, orimg_w - (pad_w - pad_w // 2), orimg_h - (pad_h - pad_h // 2))
        image = image_paded.crop(box)

        return image

    def img_slice_no_overlap(self, image, slice_size=256):
        if (image.size[0] % slice_size != 0) or (image.size[1] % slice_size != 0):
            print("Image shape error!")
            exit(-1)

        imgs = []
        imgs_w = image.size[0] // slice_size
        imgs_h = image.size[1] // slice_size

        for i in range(imgs_h):
            for j in range(imgs_w):
                boxs = [j * slice_size, i * slice_size, (j + 1) * slice_size, (i + 1) * slice_size]
                image_temp = image.crop(boxs)
                imgs.append(image_temp)

        return imgs, imgs_w, imgs_h

    def img_merge_no_overlap(self, imgs, imgs_w, imgs_h):

        slice_size = imgs[0].size[0]
        image = Image.new(imgs[0].mode, (slice_size * imgs_w, slice_size * imgs_h), (0, 0, 0))

        k = 0
        for i in range(imgs_h):
            for j in range(imgs_w):
                image.paste(imgs[k], (j * slice_size, i * slice_size))
                k += 1

        return image

    def create_trainval_list(self ,trainval_percent:float = 1,train_percent:float = 0.9,if_randomseed:bool = True):
        if if_randomseed:
            random.seed(0)

        img_trainval_list_path = os.path.join(self.data_path , 'ImageSets')
        os.makedirs(img_trainval_list_path , exist_ok = True)

        trainval_intdict = random.sample(range(self.img_num) , int(self.img_num * trainval_percent))
        train_intdict = random.sample(trainval_intdict, int(len(trainval_intdict) * train_percent))

        ftrainval_list = open(os.path.join(img_trainval_list_path , 'trainval.txt') , 'w')
        ftrain_list    = open(os.path.join(img_trainval_list_path , 'train.txt')    , 'w')
        fval_list      = open(os.path.join(img_trainval_list_path , 'val.txt')      , 'w')
        ftest_list     = open(os.path.join(img_trainval_list_path , 'test.txt')     , 'w')

        for i in range(self.img_num):
            name = self.file_list[i] + '\n'
            if i in trainval_intdict:
                ftrainval_list.write(name)
                if i in train_intdict:
                    ftrain_list.write(name)
                else:
                    fval_list.write(name)
            else:
                ftest_list.write(name)

        ftrainval_list.close()
        ftrain_list.close()
        fval_list.close()
        ftest_list.close()
        print("Generating txt in ImageSets done!")
        print("train and val size:", self.img_num)
        print("train size:", int(len(trainval_intdict) * train_percent))
        print("val size:", self.img_num - int(len(trainval_intdict) * train_percent))

    def empty_clean(self):
        print("------start empty cleaning!-----")
        file_list = self.file_list
        tif_path = self.tif_path
        png_path = self.png_path
        cout = 0
        for k , file in enumerate(file_list):
            png = self.read_png(file)
            png = np.asarray(png)
            if not png.any() > 0:
                os.remove(tif_path + "/" + file + ".tif")
                os.remove(png_path + "/" + file + ".png")
                os.remove(self.data_path + "/blent_img/" + file + '.tif')
                cout += 1
            if k%100 == 0:
                print("{} images have completed!{:.2%}!".format(k , float(k/len(file_list))))
        print(cout)
        print("-----------complete!------------")

    def add_copypaste_datas(self,data_dir:str,prefix:float=10):
        prefix = "{}%".format(prefix)

        print("-----------Start Making CP Datasets------------")

        self.create_trainval_list(trainval_percent=1,train_percent=0.8)

        copy_paste_SW = SW(data_dir)

        copy_num = int(self.img_num * 1.0)
        if copy_num > copy_paste_SW.img_num:
            copy_num = copy_paste_SW.img_num
        copy_index = random.sample(range(copy_paste_SW.img_num),copy_num)

        img_trainval_list_path = os.path.join(self.data_path, 'ImageSets')
        ftrainval_list = open(os.path.join(img_trainval_list_path , 'trainval.txt') , 'a')
        ftrain_list    = open(os.path.join(img_trainval_list_path , 'train.txt')    , 'a')


        for i in copy_index:
            file = copy_paste_SW.file_list[i]
            old_img_path = copy_paste_SW.tif_path + '/' + file + ".tif"
            new_img_path = self.tif_path + '/' + prefix + '_' + file + ".tif"
            old_png_path = copy_paste_SW.png_path + '/' + file + ".png"
            new_png_path = self.png_path + '/' + prefix + '_' + file + ".png"

            shutil.copy(old_img_path,new_img_path)
            shutil.copy(old_png_path,new_png_path)

            ftrainval_list.write(prefix + '_' + file + "\n")
            ftrain_list.write(prefix + '_' + file + "\n")

            if i%400 ==0:
                print("Complete {}!".format(i))

        ftrainval_list.close()
        ftrain_list.close()

        print("add {} images of copy_pastes".format(copy_num))
        print("-----------Making CP({}) Datasets Complete------------".format(prefix))


def CP_datasets_pn_count():
    for i in [1.5,2,3,4,5,10,15,20]:
        data_path = "E:/{}%".format(i)
        sw = SW(data_path)
        print("刚完成Copy-Paste未裁切的({}%)数据集的正负样本比例是".format(i))
        sw.p_n_sample_count()

        data_path = "E:/{}%/slice_img".format(i)
        sw = SW(data_path)
        print("刚完成Copy-Paste并裁切的({}%)数据集的正负样本比例是".format(i))
        sw.p_n_sample_count()

        data_path = "E:/0409building&{}%copypaste".format(i)
        sw = SW(data_path)
        print("完成全套数据处理的CP({}%)数据集的正负样本比例是".format(i))
        sw.p_n_sample_count()

def files_copy(sourse_path:str = "" , target_path:str = ""):
    sourse_path = os.path.abspath(sourse_path)
    target_path = os.path.abspath(target_path)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if os.path.exists(sourse_path):
        shutil.rmtree(target_path)
    else:
        print("target path does not exist!")
        exit(0)

    shutil.copytree(sourse_path,target_path)
    print("copy files finished!")

if __name__ == '__main__':
    data_path = "E:/"

    sw = SW(data_path)

    sw.p_n_sample_count()



