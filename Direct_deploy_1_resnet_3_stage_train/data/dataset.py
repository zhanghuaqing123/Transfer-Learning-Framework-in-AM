import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import ImageFile, Image
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParametersDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        image_dim=(320, 320),
        pre_crop_transform=None,
        post_crop_transform=None,
        regression=False,
        flow_rate=False,
        feed_rate=False,
        z_offset=False,
        hotend=False,
        per_img_normalisation=False,
        dataset_name=None,
    ):
        self.dataframe = pd.read_csv(csv_file)
        ## 如果需要修改csv文件中的路径数据，不需要则注释掉
        # self.dataframe['img_path'] = "/home/data/ZHQ/caxton_dataset/20240125215540dataset_switch=长方体/" + self.dataframe['img_path'].str.slice(21)
        # self.dataframe.dropna(inplace=True)
        # self.dataframe.reset_index(drop=True, inplace=True)  # 重新设置索引，丢弃之前的索引

        ### 如果想要针对性的去除109和131
        # # 找到包含 'print131' 或 'print109' 的行的索引
        # indices_to_remove = self.dataframe[self.dataframe['img_path'].str.contains('print131|print109')].index
        # # 删除这些行
        # self.dataframe.drop(indices_to_remove, inplace=True)
        # self.dataframe.reset_index(drop=True, inplace=True)  # 重新设置索引，丢弃之前的索引
        # # 打印删除后的数据集大小
        # print(self.dataframe.shape)

        #### 针对性选择数据集
        # 找到包含 'print131' 或 'print109' 的行
        # self.dataframe = self.dataframe[self.dataframe['img_path'].str.contains(dataset_name)]
        # # 重置索引，因为筛选操作会保留原来的索引
        # self.dataframe.reset_index(drop=True, inplace=True)


        #### 如果加载数据集131，不需要修改已有的image路径，因此需要注释掉上述代码
        self.root_dir = root_dir
        self.pre_crop_transform = pre_crop_transform
        self.post_crop_transform = post_crop_transform

        self.image_dim = image_dim

        self.use_flow_rate = flow_rate
        self.use_feed_rate = feed_rate
        self.use_z_offset = z_offset
        self.use_hotend = hotend

        self.per_img_normalisation = per_img_normalisation

        self.targets = []

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        self.targets = []
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.img_path[idx])
        
        dim = self.image_dim[0] / 2

        left = self.dataframe.nozzle_tip_x[idx] - dim
        top = self.dataframe.nozzle_tip_y[idx] - dim
        right = self.dataframe.nozzle_tip_x[idx] + dim
        bottom = self.dataframe.nozzle_tip_y[idx] + dim

        image = Image.open(img_name)
        if self.pre_crop_transform:
            image = self.pre_crop_transform(image)
        image = image.crop((left, top, right, bottom))

        if self.per_img_normalisation:
            tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
            image = tfms(image)
            mean = torch.mean(image, dim=[1, 2])
            std = torch.std(image, dim=[1, 2])
            image = transforms.Normalize(mean, std)(image)
        else:
            if self.post_crop_transform:
                image = self.post_crop_transform(image)

        if self.use_flow_rate:
            flow_rate_class = int(self.dataframe.flow_rate_class[idx])
            self.targets.append(flow_rate_class)

        if self.use_feed_rate:
            feed_rate_class = int(self.dataframe.feed_rate_class[idx])
            self.targets.append(feed_rate_class)

        if self.use_z_offset:
            z_offset_class = int(self.dataframe.z_offset_class[idx])
            self.targets.append(z_offset_class)

        if self.use_hotend:
            hotend_class = int(self.dataframe.hotend_class[idx])
            self.targets.append(hotend_class)

        y = torch.tensor(self.targets, dtype=torch.long)
        sample = (image, y)
        return sample
