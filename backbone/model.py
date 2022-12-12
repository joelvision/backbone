import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics import Accuracy

class Classifier(LightningModule):
    def __init__(self, model, loss, optimizer, scheduler):
        super(Classifier, self).__init__()
        self.model= model
        self.loss= loss
 
        self.metric= Accuracy(num_classes= 100)
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def validation_epoch_end(self, output):
        x= torch.stack([x for x in output['val/loss']]).mean()
        self.log('val/best_loss', x, prog_bar= True)

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
    
    def _step(self, batch, step):
        image, label= batch
        logits= self(image)
        loss= self.loss(logits, label)
        pred= torch.argmax(logits, dim=1)
        metric= self.metric(pred, label)
        self.log(f'{step}/acc', metric, prog_bar= True)
        self.log(f'{step}/loss', loss, prog_bar= True)
        return loss
    
        