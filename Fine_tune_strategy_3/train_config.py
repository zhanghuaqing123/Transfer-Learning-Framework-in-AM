import os
from datetime import datetime
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torchvision import transforms


DATE = datetime.now().strftime("%Y%m%d_%H%M%S")

dataset_switch = 8

DATA_DIR = r"/data1/zhq/caxton"

if dataset_switch == 0:
    DATASET_NAME = "dataset_single_layer"
    DATA_CSV = os.path.join(
        DATA_DIR,
        "caxton_dataset/caxton_dataset_filtered_single.csv",
    )
    DATASET_MEAN = [0.16853632, 0.17632364, 0.10495131]
    DATASET_STD = [0.05298341, 0.05527821, 0.04611006]
elif dataset_switch == 1:
    DATASET_NAME = "print132"
    DATA_CSV = os.path.join(
        DATA_DIR,
        "caxton_dataset/caxton_dataset_filtered.csv",
        # "caxton_dataset/caxton_dataset_final.csv",
    )
    DATASET_MEAN = [0.2915257, 0.27048784, 0.14393276]
    DATASET_STD = [0.066747, 0.06885352, 0.07679665]
elif dataset_switch == 2:
    DATASET_NAME = "dataset_equal"
    DATA_CSV = os.path.join(
        DATA_DIR,
        "caxton_dataset/caxton_dataset_filtered_equal.csv",
    )
    DATASET_MEAN = [0.2925814, 0.2713622, 0.14409496]
    DATASET_STD = [0.0680447, 0.06964592, 0.0779964]
elif dataset_switch == 3:
    DATASET_NAME = "my_data_0"
    DATA_CSV = os.path.join(
        DATA_DIR,
        "caxton_dataset/20240125215540dataset_switch=长方体/print_log_filtered_classification3_gaps.csv",
    )
    # 全局mean和std
    DATASET_MEAN = [0.2915257, 0.27048784, 0.14393276]
    DATASET_STD = [0.066747, 0.06885352, 0.07679665]
    # 局部mean和std
    # DATASET_MEAN = [0.32622337,0.32254382,0.29839558]
    # DATASET_STD = [0.26357063,0.2545051, 0.25629888]

elif dataset_switch == 4:
    DATASET_NAME = "print131"
    DATA_CSV = os.path.join(
        DATA_DIR,
        "caxton_dataset/print131/print_log_filtered_classification3_gaps.csv",
    )
    # 全局mean和std
    DATASET_MEAN = [0.2915257, 0.27048784, 0.14393276]
    DATASET_STD = [0.066747, 0.06885352, 0.07679665]
    # # 局部mean和std
    # DATASET_MEAN = [0.33483452, 0.11479522, 0.01575311]
    # DATASET_STD = [0.12662945, 0.07536672, 0.06436251]

elif dataset_switch == 5:
    DATASET_NAME = "print109"
    DATA_CSV = os.path.join(
        DATA_DIR,
        "caxton_dataset/print109/print_log_filtered_classification3_gaps.csv",
    )
    DATASET_MEAN = [0.2915257, 0.27048784, 0.14393276]
    DATASET_STD = [0.066747, 0.06885352, 0.07679665]

elif dataset_switch == 6:
    DATASET_NAME = "print132"
    DATA_CSV = os.path.join(
        DATA_DIR,
        "caxton_dataset/print132/print_log_filtered_classification3_gaps.csv",
    )
    DATASET_MEAN = [0.2915257, 0.27048784, 0.14393276]
    DATASET_STD = [0.066747, 0.06885352, 0.07679665]

elif dataset_switch == 7:
    DATASET_NAME = "print84"
    DATA_CSV = os.path.join(
        DATA_DIR,
        "caxton_dataset/print84/print_log_filtered_classification3_gaps.csv",
    )
    DATASET_MEAN = [0.2915257, 0.27048784, 0.14393276]
    DATASET_STD = [0.066747, 0.06885352, 0.07679665]

elif dataset_switch == 8:
    DATASET_NAME = "print109_to_132"
    source_csv_file = os.path.join(
        DATA_DIR,
        "caxton_dataset/print132/print_log_filtered_classification3_gaps.csv",
    )
    target_csv_file = os.path.join(
        DATA_DIR,
        "caxton_dataset/print109/print_log_filtered_classification3_gaps.csv",
    )

    DATASET_MEAN = [0.2915257, 0.27048784, 0.14393276]
    DATASET_STD = [0.066747, 0.06885352, 0.07679665]
    # DATASET_MEAN = [0.33483452, 0.11479522, 0.01575311]
    # DATASET_STD = [0.12662945, 0.07536672, 0.06436251]
INITIAL_LR = 0.0002 # 小学习率
# INITIAL_LR = 0.001 # 大学习率

BATCH_SIZE = 60
MAX_EPOCHS = 15
NUM_WORKERS = 8
NUM_NODES = 1
NUM_GPUS = 1
TRANFER_Weight = 5 # 迁移学习权重，0.1,0.5,1,2,5,10
ONly_source = True# True只加载训练集数据，并对该数据做训练集和测试集划分，False：同时加载源域和目标域数据
LR_shelduler_monitor = "val_loss"  # "val_loss" or "train_loss",决定学习率调整的监控指标（验证损失或者训练损失）
TRansform = False
# ACCELERATOR = "ddp"
# ACCELERATOR = "ddp_find_unused_parameters_true"
ACCELERATOR = "auto"


def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_dirs(path):
    try:
        os.makedirs(path)
    except:
        pass

preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.2915257, 0.27048784, 0.14393276],
            [0.2915257, 0.27048784, 0.14393276],
        )
    ],
)