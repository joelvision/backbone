import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics import Accuracy

class Classifier(LightningModule):
    def __init__(self, model, loss, optimizer, scheduler):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.model= model
        self.loss= loss
 
        self.train_acc= Accuracy(task='multiclass', num_classes= 100)
        self.val_acc= Accuracy(task='multiclass', num_classes= 100)
        self.test_acc= Accuracy(task='multiclass', num_classes= 100)
        
        self.train_loss= MeanMetric()
        self.val_loss= MeanMetric()
        self.test_loss= MeanMetric()
        
        self.val_loss_best= MeanMetric()
        
    def forward(self, x):
        return self.model(x)
    
    def on_train_start(self):
        self.val_loss_best.reset()
        
    def training_step(self, batch, batch_idx):
        loss, pred, label= self._step(batch)
        
        self.train_loss(loss)
        self.train_acc(pred, label)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": pred, "target": label}

    def validation_step(self, batch, batch_idx):
        loss, pred, label= self._step(batch)
        
        self.val_loss(loss)
        self.val_acc(pred, label)
        self.log("train/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": pred, "target": label}
    
    def validation_epoch_end(self, output):
        loss= self.val_loss.compute()
        self.val_loss_best(loss)
        loss= self.val_loss_best.compute()
        
        self.log('val/loss_vest', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer= self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def _step(self, batch):
        image, label= batch
        logits= self(image)
        loss= self.loss(logits, label)
        pred= torch.argmax(logits, dim=1)

        return loss, pred, label
    
        