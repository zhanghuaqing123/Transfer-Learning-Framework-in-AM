import os

from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from torchvision import transforms
import torch
from PIL import ImageFile
import numpy as np
from .dataset import ParametersDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParametersDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        data_dir,
        source_csv_file,
        target_csv_file,
        dataset_name,
        mean,
        std,
        load_saved=False,
        transform=True,
        image_dim=(320, 320),
        per_img_normalisation=False,
        flow_rate=True,
        feed_rate=True,
        z_offset=True,
        hotend=True,
        number_workers=8,
        drop_last=True,
        only_source=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.source_csv_file = source_csv_file
        self.target_csv_file = target_csv_file
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.mean = mean
        self.std = std
        self.transform = transform
        self.num_workers = number_workers
        self.drop_last = drop_last
        self.only_source = only_source
        if self.transform:
            self.pre_crop_transform = transforms.Compose(
                [
                    transforms.RandomRotation(10),
                    transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
                ]
            )
            self.post_crop_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, hue=0.1, saturation=0.1
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.pre_crop_transform = None
            self.post_crop_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
        self.dims = (3, 224, 224)
        self.num_classes = 3
        self.load_saved = load_saved
        self.image_dim = image_dim
        self.per_img_normalisation = per_img_normalisation

        self.use_flow_rate = flow_rate
        self.use_feed_rate = feed_rate
        self.use_z_offset = z_offset
        self.use_hotend = hotend
        self.train_dataset_num = None
        self.val_dataset_num = None
        self.save_hyperparameters()

    # def split_train_valid(self, ds):
    #     ds_len = len(ds)
    #     valid_ds_len = int(ds_len * self.valid_ratio)
    #     train_ds_len = ds_len - valid_ds_len
    #     return random_split(ds, [train_ds_len, valid_ds_len])
    # def prepare_data(self):
    #     # download only
    #     pass
    def setup(self, stage=None, save=False, test_all=False):
        # Assign train/val datasets for use in dataloaders

        # self.test_tgt_ds = ParametersDataset(
        #     csv_file=self.target_csv_file,
        #     root_dir=self.data_dir,
        #     image_dim=self.image_dim,
        #     pre_crop_transform=self.pre_crop_transform,
        #     post_crop_transform=self.post_crop_transform,
        #     flow_rate=self.use_flow_rate,
        #     feed_rate=self.use_feed_rate,
        #     z_offset=self.use_z_offset,
        #     hotend=self.use_hotend,
        #     per_img_normalisation=self.per_img_normalisation,
        # )

        if self.only_source: # 只加载源域数据
            self.train_src_ds = ParametersDataset(
                csv_file=self.source_csv_file,
                root_dir=self.data_dir,
                image_dim=self.image_dim,
                pre_crop_transform=self.pre_crop_transform,
                post_crop_transform=self.post_crop_transform,
                flow_rate=self.use_flow_rate,
                feed_rate=self.use_feed_rate,
                z_offset=self.use_z_offset,
                hotend=self.use_hotend,
                per_img_normalisation=self.per_img_normalisation,
                dataset_name=self.dataset_name,
            )
            train_size, val_size = int(0.7 * len(self.train_src_ds)), int(
                0.2 * len(self.train_src_ds)
            )
            test_size = len(self.train_src_ds) - train_size - val_size
            # 在进行第一次数据集划分之前设置随机种子
            torch.manual_seed(114514)  # seed_value是你选择的随机种子值
            self.train_src_ds, self.train_tgt_ds, self.test_dataset = torch.utils.data.random_split(self.train_src_ds, [train_size, val_size, test_size])
        else:
            self.train_src_ds = ParametersDataset(
                csv_file=self.source_csv_file,
                root_dir=self.data_dir,
                image_dim=self.image_dim,
                pre_crop_transform=self.pre_crop_transform,
                post_crop_transform=self.post_crop_transform,
                flow_rate=self.use_flow_rate,
                feed_rate=self.use_feed_rate,
                z_offset=self.use_z_offset,
                hotend=self.use_hotend,
                per_img_normalisation=self.per_img_normalisation,
            )
            self.train_tgt_ds = ParametersDataset(
                csv_file=self.target_csv_file,
                root_dir=self.data_dir,
                image_dim=self.image_dim,
                pre_crop_transform=self.pre_crop_transform,
                post_crop_transform=self.post_crop_transform,
                flow_rate=self.use_flow_rate,
                feed_rate=self.use_feed_rate,
                z_offset=self.use_z_offset,
                hotend=self.use_hotend,
                per_img_normalisation=self.per_img_normalisation,
            )
            train_size, val_size = int(0.8 * len(self.train_src_ds)), int(0.2 * len(self.train_src_ds))
            self.target_train, self.target_val = torch.utils.data.random_split(self.train_tgt_ds, [train_size, val_size])
            self.train_dataset_num = len(self.train_src_ds)
            self.val_dataset_num = len(self.train_tgt_ds)

        # if save:
        #     (
        #         self.train_dataset,
        #         self.val_dataset,
        #         self.test_dataset,
        #     ) = torch.utils.data.random_split(
        #         self.dataset, [train_size, val_size, test_size]
        #     )
        #     try:
        #         os.makedirs("data/{}/".format(self.dataset_name))
        #     except:
        #         pass
        #     torch.save(self.train_dataset, "data/{}/train.pt".format(self.dataset_name))
        #     torch.save(self.val_dataset, "data/{}/val.pt".format(self.dataset_name))
        #     torch.save(self.test_dataset, "data/{}/test.pt".format(self.dataset_name))

        # if stage == "fit" or stage is None:
            # if self.load_saved:
            #     self.train_dataset, self.val_dataset = torch.load(
            #         "data/{}/train.pt".format(self.dataset_name)
            #     ), torch.load("data/{}/val.pt".format(self.dataset_name))
            # else:
            #     self.train_dataset, self.val_dataset, _ = torch.utils.data.random_split(
            #         self.dataset, [train_size, val_size, test_size]
            #     )
            # pass
            # self.train_dataset =
        # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     if self.load_saved:
        #         self.test_dataset = torch.load(
        #             "data/{}/test.pt".format(self.dataset_name)
        #         )
        #         # print(f"Loaded test dataset length:{len(self.test_dataset)}")
        #     else:
        #         if test_all:
        #             self.test_dataset = self.dataset
        #         else:
        #             _, _, self.test_dataset = torch.utils.data.random_split(
        #                 self.dataset, [train_size, val_size, test_size]
        #             )
        #     pass
    def train_dataloader(self):
        # return DataLoader(
        #     self.train_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=8,
        #     pin_memory=True,
        # )
        if self.only_source:
            return DataLoader(
                self.train_src_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            dataset_size = len(self.target_train)
            indices = list(range(dataset_size))
            # 如果您希望在每个epoch中加载总数据集的一部分，请在这里定义那个部分的大小。
            # 例如，如果希望加载总数据集的50%，则batch_subset_size应该是dataset_size的50%。
            batch_subset_size = int(np.floor(0.2 * dataset_size))
            # PyTorch Lightning将在每个epoch开始时调用这个方法，从而产生不同的子集。
            np.random.shuffle(indices)
            subset_indices = indices[:batch_subset_size]
            # 使用SubsetRandomSampler来从数据集中随机抽取子集
            sampler = SubsetRandomSampler(subset_indices)
            # 使用sampler来指定DataLoader如何从数据集中抽取样本
            src = DataLoader(
                self.target_train,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=self.drop_last
            )
            # 为目标域执行相同的操作
            dataset_size = len(self.train_tgt_ds)
            indices = list(range(dataset_size))
            batch_subset_target_size = batch_subset_size
            np.random.shuffle(indices)
            subset_indices = indices[:batch_subset_target_size]
            sampler = SubsetRandomSampler(subset_indices)
            tgt = DataLoader(
                self.train_tgt_ds,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=self.drop_last
            )
            iterables = {'src': src, 'tgt': tgt}
            combined_loader = CombinedLoader(iterables, 'max_size_cycle')

            return combined_loader
        # src = DataLoader(self.train_src_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,drop_last=self.drop_last)
        # tgt = DataLoader(self.train_tgt_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,  pin_memory=True,drop_last=self.drop_last)
        # # self.train_dataset_num = len(self.train_src_ds)
        # # self.val_dataset_num = len(self.train_tgt_ds)
        # # print(f"train_dataloader_source:{len(src)}")
        # # print(f"train_dataloader_source:{len(tgt)}")
        # ### 测试一种新的数据加载方式
        # iterables = {'src': src,
        #              'tgt': tgt}
        # combined_loader = CombinedLoader(iterables, 'max_size_cycle')
        # return combined_loader
        # return [src,tgt]
    def val_dataloader(self):
        # return DataLoader(
        #     self.val_dataset,
        #     batch_size=self.batch_size,
        #     num_workers=8,
        #     pin_memory=True,
        # )
        if self.only_source:
            val_dataloader = DataLoader(self.train_tgt_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True, drop_last=not self.drop_last)
        else:
            val_dataloader = DataLoader(self.target_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True, drop_last=not self.drop_last)
        # print(f"val_dataloader:{len(val_dataloader)}")
        return val_dataloader

    def test_dataloader(self):
        # return DataLoader(
        #     self.test_dataset,
        #     batch_size=self.batch_size,
        #     num_workers=8,
        #     pin_memory=True,
        # )
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True, drop_last=not self.drop_last)
        # print(f"test_dataloader:{len(test_dataloader)}")
        return test_dataloader
    #
    # def predict_dataloader(self):
    #     return DataLoader(self.test_tgt_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def cal_max_iter(self):
        n_iter = max(self.train_dataset_num, self.val_dataset_num)
        print(f"n_iter:{n_iter}, n_batch:{n_iter/self.batch_size}")
        return int(n_iter/self.batch_size)


