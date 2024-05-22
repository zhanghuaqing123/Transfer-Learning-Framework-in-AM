import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .backbone import network_dict
from .residual_attention_network import (
    ResidualAttentionModel_56 as ResidualAttentionModel,
)
from .Domain_Discriminator import DomainClassifier
import pytorch_lightning as pl
import numpy as np
import csv
from datetime import datetime
import pandas as pd
import os
# from pytorch_lightning import *
import torchmetrics
from .mmd_loss import MMDLoss
class ParametersClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr=1e-3,
        transfer=False,
        trainable_layers=1,
        gpus=1,
        retrieve_layers=True,
        retrieve_masks=False,
        test_overwrite_filename=False,
        gamma = 10,
        max_iter_per_epoch = 1000,
        trsnfer_weight = 0.01,
        lr_shelduler_monitor = "val_loss",
        max_epoch=50,
        batchsize=60,
        base_net="resnet50",
        log_dir = "./logs",
    ):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.max_iter_per_epoch = max_iter_per_epoch
        self.max_epochs = max_epoch
        self.gpu_nums = gpus
        self.trsnfer_weight = trsnfer_weight
        self.batchsize = batchsize
        self.acc_log_dir = log_dir
        # self.accuracy_csv = os.path.join(self.acc_log_dir, 'accuracy.csv')
        # 在训练开始时创建文件并写入表头
        # if self.trainer.is_global_zero:  # Only do this for one process in case of distributed training
        # os.makedirs(os.path.dirname(self.accuracy_csv), exist_ok=True)
        # with open(self.accuracy_csv, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['epoch', 'train_acc', 'val_acc'])

        self.criterion_dc = nn.CrossEntropyLoss() # DANN的域分类损失
        self.__dict__.update(locals())
        # self.attention_model = ResidualAttentionModel(
        #     retrieve_layers=retrieve_layers, retrieve_masks=retrieve_masks
        # )
        self.attention_model = network_dict[base_net]()
        bottleneck_list = [nn.Linear(2048, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        self.dc = DomainClassifier() # 域分类器
        self.mmd_loss = MMDLoss()
        num_ftrs = self.attention_model.output_num() # 2048
        self.attention_model.fc = nn.Identity() # 2048
        self.fc1 = nn.Linear(num_ftrs, num_classes) # 2048直接到3
        self.fc2 = nn.Linear(num_ftrs, num_classes)
        self.fc3 = nn.Linear(num_ftrs, num_classes)
        self.fc4 = nn.Linear(num_ftrs, num_classes)

        if transfer:
            for child in list(self.attention_model.children())[:-trainable_layers]:
                for param in child.parameters():
                    param.requires_grad = False
        self.save_hyperparameters()

        # self.train_acc = pl.metrics.Accuracy()
        # self.train_acc0 = pl.metrics.Accuracy()
        # self.train_acc1 = pl.metrics.Accuracy()
        # self.train_acc2 = pl.metrics.Accuracy()
        # self.train_acc3 = pl.metrics.Accuracy()
        # self.val_acc = pl.metrics.Accuracy()
        # self.val_acc0 = pl.metrics.Accuracy()
        # self.val_acc1 = pl.metrics.Accuracy()
        # self.val_acc2 = pl.metrics.Accuracy()
        # self.val_acc3 = pl.metrics.Accuracy()
        # self.test_acc = pl.metrics.Accuracy()

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_acc0 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_acc1 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_acc2 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_acc3 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc0 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc1 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc2 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc3 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)


        self.name = "ResidualAttentionClassifier"
        self.retrieve_layers = retrieve_layers
        self.retrieve_masks = retrieve_masks
        self.gpus = gpus
        self.lr_shelduler_monitor = lr_shelduler_monitor
        self.sync_dist = True if self.gpus > 1 else False
        self.test_overwrite_filename = test_overwrite_filename
        self.test_step_outputs = []

    def forward(self, X):
        X = self.attention_model(X) # X是个元组，可以被顺序访问，其中第1个元素是尚未经过线性层的数据，其feature map：2048
        if self.retrieve_layers or self.retrieve_masks: # 需要返回中间层，则使用retrieve_layers=True并且retrieve_masks=false
            # feature_map = self.bottleneck_layer(X[0])
            # out1 = self.fc1(feature_map)
            # out2 = self.fc2(feature_map)
            # out3 = self.fc3(feature_map)
            # out4 = self.fc4(feature_map)
            # out1 = self.fc1(X[0])
            # out2 = self.fc2(X[0])
            # out3 = self.fc3(X[0])
            # out4 = self.fc4(X[0])
            out1 = self.fc1(X[-1])
            out2 = self.fc2(X[-1])
            out3 = self.fc3(X[-1])
            out4 = self.fc4(X[-1])
            return (out1, out2, out3, out4), X  # 其中X是个元组，可以被顺序访问，其中倒数第2个元素是尚未经过线性层的数据，其feature map：2048
        out1 = self.fc1(X)
        out2 = self.fc2(X)
        out3 = self.fc3(X)
        out4 = self.fc4(X)
        return (out1, out2, out3, out4)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, threshold=0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.lr_shelduler_monitor,
        }

    def get_alpha(self):
        return 2. / (1. + np.exp(-self.gamma * self.global_step*self.gpu_nums / (self.max_iter_per_epoch* self.max_epochs))) - 1

    def compute_dc_loss(self, out_layer_s,out_layer_t,y_hat_s, y_hat_t):
        out_layer_s = self.bottleneck_layer(out_layer_s) # 2048-256
        out_layer_t = self.bottleneck_layer(out_layer_t) # 2048-256
        alpha = self.get_alpha()
        y_hat_dc = self.dc(torch.cat([out_layer_s, out_layer_t]), alpha) # 256-256-2
        y_dc = torch.cat([torch.zeros_like(y_hat_s[:, 0]), torch.ones_like(y_hat_t[:, 0])]).long()
        loss_dc = self.criterion_dc(y_hat_dc, y_dc)
        return loss_dc, alpha

    def compute_mmd_loss(self, embed_s, embed_t):
        all_loss = 0
        for i in range(len(embed_s)):
            all_loss += 0.2*self.mmd_loss(embed_s[i].view(embed_s[i].size()[0], -1), embed_t[i].view(embed_t[i].size()[0], -1))
        # all_loss = all_loss + 0.2*self.mmd_loss(embed_s[-1], embed_t[-1]) # 压缩前做一次
        # out_layer_s = self.bottleneck_layer(embed_s[-1]) # 2048-256
        # out_layer_t = self.bottleneck_layer(embed_t[-1]) # 2048-256
        # all_loss = all_loss + 0.2*self.mmd_loss(out_layer_s, out_layer_t) # 压缩后做一次
        return all_loss,len(embed_s)

    # def get_alpha(self):
    #     return 2. / (1. + np.exp(-self.gamma * self.global_step / (self.num_step * self.max_epochs))) - 1

    def training_step(self, train_batch, batch_idx):
        # x, y = train_batch
        # y_hats = self.forward(x)
        # y_hat0, y_hat1, y_hat2, y_hat3 = y_hats
        # y = y.t()
        #
        # _, preds0 = torch.max(y_hat0, 1)
        # loss0 = F.cross_entropy(y_hat0, y[0])
        #
        # _, preds1 = torch.max(y_hat1, 1)
        # loss1 = F.cross_entropy(y_hat1, y[1])
        #
        # _, preds2 = torch.max(y_hat2, 1)
        # loss2 = F.cross_entropy(y_hat2, y[2])
        #
        # _, preds3 = torch.max(y_hat3, 1)
        # loss3 = F.cross_entropy(y_hat3, y[3])
        #
        # # loss = loss0 + loss1 + loss2 + loss3
        # loss = loss0 + loss1 + loss2 + loss3 # 如果只考虑一个任务的loss
        # preds = torch.stack((preds0, preds1, preds2, preds3))
        ### 三种加载方式之一，使用combin时
        # (x_s, y_s)= train_batch[0]["src"]
        # (x_t, y_t)= train_batch[0]["tgt"]
        ### 三种加载方式之一，使用list时
        # (x_s, y_s),(x_t,y_t) = train_batch
        ### 三种加载方式之一，使用单个数据集时
        x_s, y_s = train_batch
        y_hat_s, tuple_s = self.forward(x_s) # 输出的是一个元组，可以被顺序访问，其中第1个元素是尚未经过线性层的数据，其feature map：2048
        # y_hat_t, tuple_t = self.forward(x_t)
        # out_layer_s, embed_s = tuple_s[-1],tuple_s
        # out_layer_t, embed_t = tuple_t[-1],tuple_t
        y_hat0, y_hat1, y_hat2, y_hat3 = y_hat_s # 仅仅使用源域的数据来分类
        y_s = y_s.t()

        _, preds0 = torch.max(y_hat0, 1)
        loss0 = F.cross_entropy(y_hat0, y_s[0])

        _, preds1 = torch.max(y_hat1, 1)
        loss1 = F.cross_entropy(y_hat1, y_s[1])

        _, preds2 = torch.max(y_hat2, 1)
        loss2 = F.cross_entropy(y_hat2, y_s[2])

        _, preds3 = torch.max(y_hat3, 1)
        loss3 = F.cross_entropy(y_hat3, y_s[3])
        loss_cls = loss0 + loss1 + loss2 + loss3
        preds = torch.stack((preds0, preds1, preds2, preds3))

        # self.loss_func = AdversarialLoss(gamma=self.gamma, max_iter=self.max_iter)
        # loss_adv, accuracy = self.loss_func(embed_s[-2], embed_t[-2])
        # loss_adv, alpha = self.compute_dc_loss(out_layer_s, out_layer_t, y_hat_s[0], y_hat_t[0])
        # loss_adv, alpha = self.compute_mmd_loss(tuple_s, tuple_t)
        # loss = loss_cls + self.trsnfer_weight * loss_adv
        loss = loss_cls
        ## 计算域损失

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "loss_cls",
            loss_cls,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )

        # self.log(
        #     "loss_adv",
        #     loss_adv,
        #     on_epoch=True,
        #     logger=True,
        #     sync_dist=self.sync_dist,
        #     reduce_fx="mean",
        #     batch_size=self.batchsize,
        # )
        self.log(
            "train_loss0",
            loss0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "train_loss1",
            loss1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "train_loss2",
            loss2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "train_loss3",
            loss3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )

        # self.log(
        #     "alpha",
        #     alpha,
        #     on_epoch=True,
        #     on_step=True,
        #     logger=True,
        #     sync_dist=self.sync_dist,
        #     # reduce_fx="mean",
        #     batch_size=1,
        # )
        # self.log(
        #     "accuracy_Domain",
        #     accuracy,
        #     on_epoch=True,
        #     logger=True,
        #     sync_dist=self.sync_dist,
        #     reduce_fx="mean",
        # )

        self.train_acc(preds, y_s)
        self.train_acc0(preds0, y_s[0])
        self.train_acc1(preds1, y_s[1])
        self.train_acc2(preds2, y_s[2])
        self.train_acc3(preds3, y_s[3])
        # self.train_step_outputs.append(self.train_acc(preds, y_s))
        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "train_acc0",
            self.train_acc0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "train_acc1",
            self.train_acc1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "train_acc2",
            self.train_acc2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "train_acc3",
            self.train_acc3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hats,_ = self.forward(x)
        y_hat0, y_hat1, y_hat2, y_hat3 = y_hats
        y = y.t()

        _, preds0 = torch.max(y_hat0, 1)
        loss0 = F.cross_entropy(y_hat0, y[0])

        _, preds1 = torch.max(y_hat1, 1)
        loss1 = F.cross_entropy(y_hat1, y[1])

        _, preds2 = torch.max(y_hat2, 1)
        loss2 = F.cross_entropy(y_hat2, y[2])

        _, preds3 = torch.max(y_hat3, 1)
        loss3 = F.cross_entropy(y_hat3, y[3])

        loss = loss0 + loss1 + loss2 + loss3
        preds = torch.stack((preds0, preds1, preds2, preds3))

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "val_loss0",
            loss0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "val_loss1",
            loss1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "val_loss2",
            loss2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "val_loss3",
            loss3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )

        self.val_acc(preds, y)
        self.val_acc0(preds0, y[0])
        self.val_acc1(preds1, y[1])
        self.val_acc2(preds2, y[2])
        self.val_acc3(preds3, y[3])

        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "val_acc0",
            self.val_acc0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "val_acc1",
            self.val_acc1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "val_acc2",
            self.val_acc2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.log(
            "val_acc3",
            self.val_acc3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hats,_ = self.forward(x)
        y_hat0, y_hat1, y_hat2, y_hat3 = y_hats
        y = y.t()

        _, preds0 = torch.max(y_hat0, 1)
        loss0 = F.cross_entropy(y_hat0, y[0])

        _, preds1 = torch.max(y_hat1, 1)
        loss1 = F.cross_entropy(y_hat1, y[1])

        _, preds2 = torch.max(y_hat2, 1)
        loss2 = F.cross_entropy(y_hat2, y[2])

        _, preds3 = torch.max(y_hat3, 1)
        loss3 = F.cross_entropy(y_hat3, y[3])

        loss = loss0 + loss1 + loss2 + loss3

        self.log("test_loss0", loss0)
        self.log("test_loss1", loss1)
        self.log("test_loss2", loss2)
        self.log("test_loss3", loss3)

        preds = torch.stack((preds0, preds1, preds2, preds3))
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        self.test_acc(preds, y)

        self.log(
            "test_acc",
            self.test_acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
            batch_size=self.batchsize,
        )
        # self.test_step_outputs.append(loss)
        return {"loss": loss, "preds": preds, "targets": y}

    # def test_epoch_end(self, outputs):
    #     preds = [output["preds"] for output in outputs]
    #     targets = [output["targets"] for output in outputs]
    #
    #     preds = torch.cat(preds, dim=1)
    #     targets = torch.cat(targets, dim=1)
    #
    #     os.makedirs("test/", exist_ok=True)
    #     if self.test_overwrite_filename:
    #         torch.save(preds, "test/preds_test.pt")
    #         torch.save(targets, "test/targets_test.pt")
    #     else:
    #         date_string = datetime.now().strftime("%H-%M_%d-%m-%y")
    #         torch.save(preds, "test/preds_{}.pt".format(date_string))
    #         torch.save(targets, "test/targets_{}.pt".format(date_string))
    # def on_train_epoch_end(self):
    #     if self.trainer.is_global_zero:
    #         epoch_average = self.train_acc.compute().detach().cpu().item()
    #         self.log("train_epoch_average", epoch_average)
    #         with open(self.accuracy_csv, 'a', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerow([self.current_epoch, epoch_average, ''])
            # self.test_step_outputs.clear()
    # def on_validation_epoch_end(self):
        # if self.trainer.is_global_zero:
        #     epoch_average = self.val_acc.compute().detach().cpu().item()
        #     self.log("val_epoch_average", epoch_average)
        #     # 首先，读取CSV文件中的所有内容
        #     with open(self.accuracy_csv, 'r', newline='') as file:
        #         reader = csv.reader(file)
        #         data = list(reader)
        #
        #     # 更新最后一行（当前epoch），添加验证精度
        #     data[-1][-1] = epoch_average
        #     # 重新写入CSV文件
        #     with open(self.accuracy_csv, 'w', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerows(data)
            # self.test_step_outputs.clear()
    # def on_test_epoch_end(self):
    #     epoch_average = torch.stack(self.test_step_outputs).mean()
    #     self.log("test_epoch_average", epoch_average)
    #     self.test_step_outputs.clear()  # free memory
