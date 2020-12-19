#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import torch.nn as nn

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'food.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}
    
    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
    
    def epoch_end(self, epoch, epochs, result):
        print("Epoch: [{}/{}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, epochs, result["lrs"][-1], result["train_loss"], result["val_loss"], result["val_acc"]))

class Classifier(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet18(pretrained=True)
        number_of_features =  self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 101)
        
    def forward(self, xb):
        return self.network(xb)
        
    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad=False
        for param in self.network.fc.parameters():
            param.requires_grad=True
        
    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad=True

if __name__ == '__main__':
    main()
