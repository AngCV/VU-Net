import random
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf

colors = sum([[0, 0, 0],[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
                    [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                    [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
                    [128, 64, 12]],[])
class Augmentation:
    def __init__(self):
        pass

    def rotate(self, image, mask, angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-180, 180])  # -180~180随机选一个角度旋转
        if isinstance(angle, list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        return image, mask

    def hflip(self, image, mask):  # 垂直翻转
        image = tf.hflip(image)
        mask = tf.hflip(mask)
        return image, mask

    def vflip(self, image, mask):  # 水平翻转
        image = tf.vflip(image)
        mask = tf.vflip(mask)
        return image, mask

    def randomResizeCrop(self, image, mask, scale=(0.3, 1.0), ratio=(1, 1)):
        w_image, h_image = image.size
        resize_size = h_image
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size)
        return image, mask

    def adjustContrast(self, image, mask):
        factor = transforms.RandomRotation.get_params([0, 10])  # 这里调增广后的数据的对比度
        image = tf.adjust_contrast(image, factor)
        return image, mask

    def adjustBrightness(self, image, mask):
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据亮度
        image = tf.adjust_brightness(image, factor)
        return image, mask

    def adjustSaturation(self, image, mask):  # 调整饱和度
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据亮度
        image = tf.adjust_saturation(image, factor)
        return image, mask


def augmentationData(image:Image.Image, mask:Image.Image ):

    option = np.random.choice(np.arange(0,5), size=np.random.randint(0, 4), replace=False)

    aug = Augmentation()
    for i in option:
        if i == 1:
            image, mask = aug.rotate(image, mask)
        if i == 2:
            image, mask = aug.hflip(image, mask)
        if i == 4:
            image, mask = aug.vflip(image, mask)
        if i == 3:
            image, mask = aug.adjustContrast(image, mask)

    image = tf.to_tensor(image)
    mask  = tf.to_tensor(mask)

    one = torch.ones_like(mask)
    zero = torch.zeros_like(mask)
    mask = torch.where(mask > 0, one, zero)

    image = tf.normalize(image,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return image , mask

if __name__ == '__main__':
    from SW_datasets_utils import SW
    sw = SW("data/slice_img")
    path = 'C:/Users/hasee/Desktop/test/pytorch/deeplabv3-plus-pytorch-main/logs/aug_check'
    os.makedirs(path , exist_ok = True)
    trans = transforms.ToPILImage()

    aug = Augmentation()

    a = sw.read_img(sw.file_list[1])
    b = sw.read_png(sw.file_list[1])
    b.show()
    _, c = aug.rotate(a,b)
    c = tf.to_tensor(c)

    print(c.sum())
    d = torch.ceil(c)
    print(d.sum())

    sw.mask_show(d)

    _, e = augmentationData(a, b, 1)

    sw.mask_show(e)



