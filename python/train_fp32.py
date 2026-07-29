import os

import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 0
LR = 1e-3
EPOCHS = 10

HERE = os.path.dirname(os.path.abspath(__file__))
def here(name):
    return os.path.join(HERE, name)

torch.manual_seed(SEED)
np.random.seed(SEED)

# CNN

class smallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False) # （28 + 2*1 - 3）/1 + 1 =28 
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1,bias=False) # (14 + 2*1 - 3)/1 + 1 = 14
        self.fc = nn.Linear(16*7*7,10,bias=False)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2) #28/2=14
        x = F.max_pool2d(F.relu(self.conv2(x)),2)# 14/2 = 7
        x = torch.flatten(x,1) # flatten CHW
        return self.fc(x)


if __name__ == '__main__':

    # Data

    tf=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((128/255,),(128/255,)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.MNIST(here('data'),train=True,download=True,transform=tf)
    test_set = datasets.MNIST(here('data'),train=False,download=True,transform=tf)   

    train_loader = DataLoader(train_set,batch_size=128,shuffle=True,num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set,batch_size=512,shuffle=False)

    ## train

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = smallCNN().to(device)

    n_all = sum(p.numel() for p in net.parameters())
    print(f'device {device} | Weights: {n_all}   Expected: 9064 ')

    #Adam
    optimizer = torch.optim.Adam(net.parameters(),lr=LR)

    for epoch in range(EPOCHS):
        net.train()
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(net(x),y)
            loss.backward()
            optimizer.step()

        net.eval()
        correct = 0
        with torch.no_grad():
            for x,y in test_loader:
                pred = net(x.to(device)).argmax(dim=1).cpu()
                correct += (pred==y).sum().item()

        print(f'epoch {epoch}   test acc {correct/100:.2f}%   loss {loss.item():.4f}')

    baseline_acc = correct/100

    # Export
    torch.save(net.state_dict(),here('baseline.pth')) #save weights

    imgs_u8 = test_set.data.numpy().astype(np.uint8)[:,None,:,:]
    labels = test_set.targets.numpy()

    np.save(here('testset_full.npy'),{'images': imgs_u8, 'labels': labels})

    indices = np.random.choice(len(imgs_u8), 200, replace=False)
    np.save(here('testset_200.npy'),{'images': imgs_u8[indices], 'labels': labels[indices]}
            )


