import torch

from base_trainers import BaseTrain

class MultiClassificationTrainer(BaseTrain):
    """
    Classification Train Class
    """

    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, epochs, device, checkpoint_dir = './checkpoints', patience=10, scheduler=None, verbose=False):   
        super().__init__(model, train_loader, val_loader, loss_fn, optimizer, epochs, device, checkpoint_dir, patience, scheduler)
        self.verbose = verbose

    def train_epoch(self):
        size = len(self.train_loader.dataset)
        self.model.train()
        train_loss = 0.0
        for batch, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            
            if self.verbose:
                if batch % 100 == 0:
                    loss, current = loss.item(), (batch) * 32 + len(inputs)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        self.history['train_loss'].append(train_loss / len(self.train_loader))

    def val_epoch(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, data in enumerate(self.val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()
        self.history['val_loss'].append(val_loss / len(self.val_loader))
