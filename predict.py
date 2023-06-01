
from unet import Unet

if __name__ == "__main__":

    model_path = "logs/SW_2023011212_82.22%-81.79%_VUNet_CP(2%)_FDLoss_1-5weight/ep1249-tIOU91.85%-vIOU81.79%-tloss0.023-vloss0.0595-tPA99.64%-vPA97.84%.pth"

    unet = Unet(model_path=model_path
                ,dataset_path="E:/slice")#
    #----------------------------------------------------------------------------------------------------------#

    #普通的预测
    while True:
        #img = input('Input image number or image name:')
        img = "j91"
        #img = 79
        # if not img in deeplab.sw.file_list:
        #     print("Image name error!")
        #     continue
        try:
            img = int(img)
        except:
            img = img
        else:
            img = unet.sw.file_list[int(img)]
            print(img)

        try:
            output = unet.detect_image(img)
        except:
            print("ERROR! Try again!")
            continue
        blend_img = unet.sw.img_blend_show(img,output["blend_png"],ifprint=True,ifsave=False)
        print("IoU:{:.5%}__准确率为：{:.5%}__精确率为：{:.5%}__召回率为：{:.5%}".format(output["iou"] , output["accuracy"]  , output["precision"]  , output["recall"] ))


        unet.sw.mask_show(output["pr_png"],ifprint=False).save("datas/save/{}_predict_IoU{:.2%}_PA{:.2%}_R{:.2%}.png".format(img,output["iou"],output["accuracy"],output["recall"]))

