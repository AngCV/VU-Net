import colorsys
from dss import SW_datasets_utils

import torch
import torch.nn.functional as F

from torch import nn
import torchvision.transforms.functional as tf
from nets.unet import Unet as unet

class Unet(object):
    _defaults = {

        "model_path"    : "logs/SW_2022062111_71.66%_unet_vgg_0409&10%_focal+dice_10-100weight/ep602-tIOU85.15%-vIOU71.66%-tloss0.206-vloss0.42-tPA99.01%-vPA97.52%.pth",

        "num_classes"   : 2,

        "backbone"      : "vgg",

        "input_shape"   : [256, 256],

        "mix_type"      : 0,

        "cuda"          : True,

    }

    def __init__(self,  model_path:str = "",dataset_path:str="",**kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        if model_path !="":
            self.model_path = model_path
        self.generate()
        if dataset_path !="":
            self.sw = SW_datasets_utils.SW(dataset_path)
        else:
            self.sw = SW_datasets_utils.SW("E:/")

    def generate(self):

        self.net = unet(num_classes=self.num_classes, backbone="vgg")

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, img):
        try:
            ori_image = self.sw.read_img(img)
            ori_png   = self.sw.read_png(img)
        except:
            print(img , ' Open Error! Try again!')
            return -1
        image       = ori_image.convert("RGB")

        image_paded , pad_w , pad_h = self.sw.img_pad_unless_nosize(image , self.input_shape[0])
        imgs , coordinate = self.sw.img_slice_overlap(image_paded , self.input_shape[0])

        pre_predict   = []

        for i in range(len(imgs)):

            with torch.no_grad():
                images = tf.to_tensor(imgs[i])
                images = tf.normalize(images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                images = images.unsqueeze(0)
                if self.cuda:
                    images = images.cuda()

                pr = self.net(images)
                pr = pr[0]

                temp_pr = F.softmax(pr, dim=0).cpu().numpy()
                pre_predict.append(temp_pr)

        pr_png = self.sw.mask_merge_overlap(pre_predict , image_paded.size , coordinate , num_classes=self.num_classes , size=self.input_shape[0])

        pr_png = self.sw.img_unpad_unless_nosize(pr_png , pad_w , pad_h)

        temp = self.sw.accuracy_count(ori_png , pr_png , ifprint=False)

        return {"blend_png":temp["predit_image"], "pr_png":pr_png, "iou":temp["iou"], "accuracy":temp["accuracy"],
                "precision":temp["precision"], "recall":temp["recall"], "par":temp["paramaters"] , "f1_score":temp["f1_score"]}

    def detect_small_image(self, img):

        try:
            ori_image = self.sw.read_img(img)
            ori_png   = self.sw.read_png(img)
        except:
            print(img , 'Open Error! Try again!')
            return -1

        image       = ori_image.convert("RGB")

        with torch.no_grad():

            images = tf.to_tensor(image)
            images = tf.normalize(images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            images = images.unsqueeze(0)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)
            pr = pr[0]
            pr_png = F.softmax(pr, dim=0).cpu().numpy()
            pr_png = pr_png.argmax(axis=0)
            pr_png = self.sw.mask_show(pr_png,ifprint=False)

        blend_png, iou, accuracy, precision, recall, par = self.sw.accuracy_count(ori_png , pr_png , ifprint=False)

        return {"blend_png":blend_png, "pr_png":pr_png, "iou":iou, "accuracy":accuracy, "precision":precision, "recall":recall, "par":par}


