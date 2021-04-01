import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .naic import NAIC
from .veri import VeRi
from .bases import ImageDataset
from .preprocessing import RandomErasing, RandomPatch
from .sampler import RandomIdentitySampler
from .RSA_Aug import RandomShiftingAugmentation
import logging
# -----------------------------------------
from .mydata import mydata

__factory = {
    'veri': VeRi,
    'naic': NAIC,
    'mydata': mydata

}


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths

from .bases import use_Gray
def make_dataloader(cfg):
    logger = logging.getLogger("reid_baseline.train")
    logger.info("pixel mean:{} pix std:{}".format(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD))

    if cfg.DATASETS.HARD_AUG:
        if cfg.INPUT.USE_RP:  # 使用random patch
            logger.info(
                "using random patch:{:.2f} using random rotate:{:.2f}".format(cfg.INPUT.RE_PROB, cfg.INPUT.RR_PROB))
            transform_list = [
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                # T.Pad(cfg.INPUT.PADDING),
                # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),  # 颜色抖动：亮度 对比度 饱和度 色调
                T.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False,
                                          fillcolor=128),  # 随机仿射变换
                RandomPatch(prob_happen=cfg.INPUT.RE_PROB, prob_rotate=cfg.INPUT.RR_PROB),  # 使用random patch
                RandomShiftingAugmentation(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ]
            train_transforms = T.Compose(transform_list)
        else:
            logger.info(
                "using random erasing:{:.2f} using random rotate:{:.2f}".format(cfg.INPUT.RE_PROB, cfg.INPUT.RR_PROB))
            train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                # T.Pad(cfg.INPUT.PADDING),
                # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),  # 颜色抖动：亮度 对比度 饱和度 色调
                T.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False,
                                          fillcolor=128),  # 随机仿射变换
                RandomShiftingAugmentation(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)  # 将训练集pixel mean作为填充值
            ])
    else:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),  # 调整到设定大小
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),  # 水平反转
            # T.Pad(cfg.INPUT.PADDING),
            # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            RandomShiftingAugmentation(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),  # 正则化
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)  # 随机擦除
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        "正常的三元组采样：16×8（ID×num）"
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        "直接随机采样，不能用在训练reID这里"
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set_green = ImageDataset(dataset.query_green + dataset.gallery_green, val_transforms)
    val_loader_green = DataLoader(
        val_set_green, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    val_set_normal = ImageDataset(dataset.query_normal + dataset.gallery_normal, val_transforms)
    val_loader_normal = DataLoader(
        val_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader_green, val_loader_normal, len(dataset.query_green), len(
        dataset.query_normal), num_classes


def make_dataloader_Pseudo(cfg):
    if cfg.DATASETS.HARD_AUG:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            T.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set_green = ImageDataset(dataset.query_green + dataset.gallery_green, val_transforms)
    val_loader_green = DataLoader(
        val_set_green, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader_green, len(dataset.query_green), num_classes, dataset, train_set, train_transforms
