### 根据test
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 一定要对应修改GPU_NUM
import argparse
import pytorch_lightning as pl
from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier

from train_config import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s", "--seed", default=1234, type=int, help="Set seed for training"
)

args = parser.parse_args()
seed = args.seed

set_seed(seed)

# model = ParametersClassifier(
#     num_classes=3,
#     lr=INITIAL_LR,
#     gpus=1,
#     transfer=False,
# )

model = ParametersClassifier.load_from_checkpoint(
    # checkpoint_path="/home/data/ZHQ/DANN_24/caxton-main/src/checkpoints/09032024/1234/MHResAttNet-dataset_equal-09032024-epoch=41-val_loss=1.57-val_acc=0.84.ckpt",
    # checkpoint_path="/home/data/ZHQ/DANN_24/caxton-main/src/checkpoints/09032024/1234/MHResAttNet-dataset_full-09032024-epoch=42-val_loss=1.61-val_acc=0.84.ckpt", # 这是我去掉131和109这两个数据集之后训练得到的模型
    checkpoint_path="/home/data/ZHQ/DANN_24/TL_framework/Fine_tune_strategy_3/checkpoints/20240419_135818/1234/MHResAttNet-print131_to_109-20240419_135818-epoch=03-val_loss=3.13-val_acc=0.68.ckpt", # 这是我去掉131和109这两个数据集之后训练得到的模型
    num_classes=3,
    lr=INITIAL_LR,
    gpus=1,
    transfer=False,
    retrieve_layers=True,
    retrieve_masks=False,
    Data_name=DATASET_NAME
)

model.eval()

data = ParametersDataModule(
    batch_size=BATCH_SIZE,
    data_dir=DATA_DIR,
    source_csv_file=source_csv_file,
    target_csv_file=target_csv_file,
    image_dim=(320, 320),
    dataset_name=DATASET_NAME,
    mean=DATASET_MEAN,
    std=DATASET_STD,
    transform=False,
    only_source=True,
)
data.setup(stage='test', save=False, test_all=True)

trainer = pl.Trainer(
    num_nodes=1,
    devices=1,
    # weights_summary=None,
    precision="16-mixed",
)

trainer.test(model, datamodule=data)