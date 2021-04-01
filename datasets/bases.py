from PIL import Image, ImageFile
import cv2
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
use_equalizeHist = False  # 直方图均衡化
use_Gray = False  # 灰度化

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            if use_Gray:
                img = Image.open(img_path).convert('L')  # uint8数据
                if use_equalizeHist:
                    img_cv = np.asarray(img)
                    cv2.equalizeHist(img_cv, img_cv)
                    img_cv = img_cv.reshape((256,128,1))
                    # channels = cv2.split(img_cv)
                    img_cv = np.concatenate((img_cv,img_cv,img_cv),axis=-1)
                    img = Image.fromarray(img_cv)
            else:
                img = Image.open(img_path).convert('RGB')  # uint8数据
                if use_equalizeHist:
                    "转换到opencv下进行彩色均衡化"
                    img_cv = np.asarray(img)
                    channels = cv2.split(img_cv)
                    cv2.equalizeHist(channels[0], channels[0])
                    cv2.equalizeHist(channels[1], channels[1])
                    cv2.equalizeHist(channels[2], channels[2])
                    cv2.merge(channels, img_cv)
                    img = Image.fromarray(img_cv)

            # RGB = cv2.cvtColor(np.asarray(img),cv2.COLOR_YCR_CB2RGB)
            # show_image = cv2.cvtColor(img_cv, cv2.COLOR_YCR_CB2RGB)
            # plt.subplot(121), plt.imshow(RGB), plt.title('S')
            # plt.subplot(122), plt.imshow(img_cv), plt.title('T')
            # plt.show()
                                    # HSV = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
                                    # YCrCb = cv2.cvtColor(img_cv, cv2.COLOR_RGB2YCrCb)
                                    # img = np.concatenate((img_cv, HSV, YCrCb), axis=2)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path.split('/')[-1].split('\\')[-1]
