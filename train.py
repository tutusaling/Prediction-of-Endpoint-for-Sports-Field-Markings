from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from datamodule import SoccerNetDataModule
from segformer import SoccerNetFinetuner
from constants import id2class_0b, class2id_0b

import torch
torch.set_float32_matmul_precision('high')


if __name__ == "__main__":
    data_module = SoccerNetDataModule(root='../SoccerNet', mask_root='../SoccerNet/mask', train_folder='train', valid_folder='valid', test_folder='test', batch_size=2)
    model = SoccerNetFinetuner(id2label=id2class_0b, label2id=class2id_0b, metrics_interval=10)

    wandb_logger = WandbLogger(project="sn-test")

    early_stop_callback = EarlyStopping(
        monitor="val_mean_iou", 
        min_delta=0.00, 
        patience=10, 
        verbose=False, 
        mode="max",
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=10, monitor="val_mean_iou")

    trainer = Trainer(
        max_epochs=200,
        callbacks=[early_stop_callback, checkpoint_callback],
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        fast_dev_run=False
    )

    trainer.fit(model, data_module)