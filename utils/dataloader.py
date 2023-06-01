import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from dss import segmentation_data_augmentation


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path ):
        super(UnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]
        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        try:
            img = Image.open(os.path.join(os.path.join(self.dataset_path, "tif_img"), name + ".tif"))
        except:
            img = Image.open(os.path.join(os.path.join(self.dataset_path, "tif_img"), name + ".jpg"))
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "png_mask"), name + ".png"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png = segmentation_data_augmentation.augmentationData(img, png)

        jpg         = np.array(jpg, np.float64)
        png         = np.array(png,dtype=np.uint8)
        png[png >= self.num_classes] = self.num_classes

        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels



