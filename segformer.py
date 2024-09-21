import torch
import torch.nn as nn
import torch.optim as optim
from transformers import SegformerForSemanticSegmentation
from lightning import LightningModule
import evaluate

class SoccerNetFinetuner(LightningModule):
    def __init__(self, id2label, label2id, metrics_interval):
        super().__init__()
        self.num_classes = len(id2label)
        self.id2label = id2label
        self.metrics_interval = metrics_interval

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b3-finetuned-ade-512-512",
            num_labels=self.num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,)
        
        self.train_miou = evaluate.load("mean_iou")
        self.val_miou = evaluate.load("mean_iou")
        self.test_miou = evaluate.load("mean_iou")

        self.val_outputs = []
        self.test_outputs = []

    
    
    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)
    

    def _common_step(self, batch):
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        return loss, predicted, masks


    def training_step(self, batch, batch_idx):
        loss, predicted, masks = self._common_step(batch)

        self.train_miou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        
        if batch_idx % self.metrics_interval == 0:
            metrics = self.train_miou.compute(
                num_labels=self.num_classes, 
                ignore_index=255, 
                reduce_labels=False,
            )
            
            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            
            for k,v in metrics.items():
                self.log(k, v, prog_bar=True, sync_dist=True)
            
            return (metrics)
        else:
            return ({'loss': loss})
    
    def validation_step(self, batch, batch_idx):
        loss, predicted, masks = self._common_step(batch)

        self.val_outputs.append(loss)
        
        self.val_miou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        return ({'val_loss': loss})
    
    def on_validation_epoch_end(self):
        metrics = self.val_miou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
        )
        
        avg_val_loss = torch.stack(self.val_outputs).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        
        metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
        for k,v in metrics.items():
            self.log(k, v, sync_dist=True)

        return metrics
    
    def test_step(self, batch, batch_idx):
        loss, predicted, masks = self._common_step(batch)
        
        self.test_outputs.append(loss)
        
        self.test_miou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        return ({'test_loss': loss})
    
    def on_test_epoch_end(self):
        metrics = self.test_miou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
       
        avg_test_loss = torch.stack(self.test_outputs).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}
        
        for k,v in metrics.items():
            self.log(k, v, sync_dist=True)
        
        return metrics

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.00006)
        return optimizer