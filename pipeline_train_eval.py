import torch
import torchmetrics
from torch import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from kanvolutionalNets import KanvolutionalNetwork, KanVolNet
from utils_training import torch_train_val_split
from trainers import MultiClassificationTrainer

transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,))
    ])

mnist_flag = False
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) if mnist_flag else datasets.FashionMNIST(root='./Fdata', train=True, download=True, transform=transform)

BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, val_loader, _ = torch_train_val_split(
    dataset,
    batch_train=BATCH_SIZE,
    batch_eval=BATCH_SIZE
    )       

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) if mnist_flag else datasets.FashionMNIST(root='./Fdata', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


model_kanv = KanVolNet(1, 10, order_p=3, order_q=2, kernel_size=(3,3), stride=1, padding=0, scale=False).to(DEVICE) #KanvolutionalNetwork(1, 5, order_p=3, order_q=2)#

### count total number of parameters ###
x = torch.randn(32, 1, 28, 28).to(DEVICE)
model_kanv(x) # we use lazy initialization, so we need to run the model once to initialize the parameters
print(sum([i.numel() for i in model_kanv.parameters() if i.requires_grad]))
###

# Training Hyperparams
epochs = 50

# Optimizer
lr = 1e-2 # lower the lr
weight_decay = 1e-5
optimizer = torch.optim.AdamW(model_kanv.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=5, gamma=0.5) # after a point stop the decrease of loss

kanv_ = MultiClassificationTrainer(
    model_kanv,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    epochs,
    DEVICE,
    scheduler=scheduler
)

hist, checkp_dir = kanv_.train_nn()

# load the best model
kanv_.load_checkpoint(checkp_dir)
# start eval
metric =  torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(DEVICE)

model_kanv.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model_kanv(data)
        preds = output.argmax(dim=1, keepdim=False)
        acc = metric(preds, target)
        print(f"Accuracy on batch: {acc}")
    
acc = metric.compute()
print(f"Accuracy on all data: {acc:.7f}")
