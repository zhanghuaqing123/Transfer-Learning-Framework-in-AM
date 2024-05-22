import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import csv
from pytorch_lightning.callbacks import Callback
from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier
from lightning.pytorch.utilities.model_summary import ModelSummary
from train_config import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 一定要对应修改GPU_NUM
torch.set_float32_matmul_precision('high') # 设置矩阵乘法精度,有三个选项：'high'、'medium'、'highest'

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--seed", default=1234, type=int, help="Set seed for training"
    )
    parser.add_argument("--TRANFER_Weight",
                        default=0.01,
                        type=float,
                        help="Path to a model checkpoint to resume training")
    parser.add_argument(
        "-e",
        "--epochs",
        default=MAX_EPOCHS,
        type=int,
        help="Number of epochs to train the model for",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        # default="/data1/zhq/TL_framework/Direct_deploy_1_resnet/checkpoints/20240420_105104/1234/MHResAttNet-131-20240420_105104-epoch=08-val_loss=1.41-val_acc=0.88.ckpt",
        # default="/data1/zhq/TL_framework/Direct_deploy_1_resnet/checkpoints/20240420_105104/1234/MHResAttNet-131-20240420_105104-epoch=10-val_loss=1.43-val_acc=0.89.ckpt", # 这是我去掉131和109这两个数据集之后训练得到的模型
        # default="/data1/zhq/TL_framework/Direct_deploy_1_resnet/checkpoints/20240420_133310/1234/MHResAttNet-132-20240420_133310-epoch=12-val_loss=2.65-val_acc=0.81.ckpt", # 这是我去掉131和109这两个数据集之后训练得到的模型
        default="/data1/zhq/TL_framework/Direct_deploy_1_resnet/checkpoints/20240511_142929/1234/MHResAttNet-109-20240511_142929-epoch=10-val_loss=1.69-val_acc=0.87.ckpt",
        # 这是我去掉131和109这两个数据集之后训练得到的模型
        # num_classes=3,
        # default="",   #不使用任何预训练模型
        type=str,
        help="Path to a model checkpoint to resume training"
    )

    args = parser.parse_args()
    return args

def main():
    args = args_parser()
    seed = args.seed

    set_seed(seed)
    logs_dir = "logs/logs-{}/{}/".format(DATE, seed)
    logs_dir_default = os.path.join(logs_dir, "default")

    make_dirs(logs_dir)
    make_dirs(logs_dir_default)

    tb_logger = pl_loggers.TensorBoardLogger(logs_dir)

    # 设置 CSVLogger
    csv_logger = pl_loggers.CSVLogger(logs_dir,name='my_model_logs')
    # 初始化logging模块

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/{}/{}/".format(DATE, seed),
        filename="MHResAttNet-{}-{}-".format(DATASET_NAME, DATE)
        + "{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=3,
        mode="min",
    )

    model = ParametersClassifier(
        num_classes=3,
        lr=INITIAL_LR,
        gpus=NUM_GPUS,
        transfer=False,
        trainable_layers=1,
        retrieve_layers=True,
        retrieve_masks=False,
        test_overwrite_filename=False,
        gamma=10,
        max_iter_per_epoch=1000,
        trsnfer_weight=0.01,
        lr_shelduler_monitor="val_loss",
        max_epoch=50,
        batchsize=60,
        base_net="resnet50",
        log_dir=logs_dir,
    )

    # Load checkpoint if path is provided,如果迁移学习，需要加载已经预训练好的模型
    if args.checkpoint_path:
        model = ParametersClassifier.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            lr=INITIAL_LR,
            gpus=NUM_GPUS,
            transfer=True,
            trainable_layers=1,
            retrieve_layers=True,
            retrieve_masks=False,
        ) # strict=False，可以加载部分模型参数
    #
    # # model.print_trainable_layers() # 打印可训练的层

    data = ParametersDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        source_csv_file=source_csv_file,
        target_csv_file=target_csv_file,
        dataset_name=DATASET_NAME,
        mean=DATASET_MEAN,
        std=DATASET_STD,
        number_workers=NUM_WORKERS,
        only_source=ONly_source,
        transform=TRansform,
    )
    # # 定义CSV文件路径
    # csv_file = "../csv_file.csv"
    #
    # # 实例化你的Callback
    # metrics_csv_logger = MetricsCSVLogger(csv_file)

    trainer = pl.Trainer(
        num_nodes=NUM_NODES,
        # gpus=NUM_GPUS,
        devices=NUM_GPUS,
        # distributed_backend=ACCELERATOR,
        strategy=ACCELERATOR,
        max_epochs=args.epochs,
        logger=[csv_logger,tb_logger],
        # weights_summary=None,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        # fast_dev_run=5,
    )

    trainer.fit(model, data)
    trainer.test(datamodule=data)  # Run test using the test dataloader
if __name__ == "__main__":
    main()

