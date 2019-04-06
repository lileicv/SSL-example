import numpy as np
from keras.datasets import mnist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 16, 5, 1)
        self.fc1 = nn.Linear(4*4*16, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout(x, 0.2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout(x, 0.2)
        x = x.view(-1, 4*4*16)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} '.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='')

def train_ssl(model, device, la_loader, un_loader, optimizer, epoch):
    model.train()

    for batch_idx,(data, _) in enumerate(un_loader):
        iter_la = iter(la_loader)
        data_la, target_la = next(iter_la)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        plabel = output.argmax(-1)
        loss = F.nll_loss(output, plabel)

        data_la = data_la.to(device)
        target_la = target_la.to(device)
        out = model(data_la)
        loss2 = F.nll_loss(out, target_la)
        loss += loss2
        loss.backward()
        optimizer.step()
        if batch_idx%10==0:
            print('\rSSL epoch: {}, Loss: {:.6f} '.format(epoch+1, loss.item()), end='')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Global Config
batchsize = 50

# Load dataset
(xtr,ytr),(xte,yte) = mnist.load_data()
xtr = np.expand_dims(xtr,1).astype('float32')/255.
xte = np.expand_dims(xte,1).astype('float32')/255.
ytr = ytr.astype(np.int)
yte = yte.astype(np.int)
perm = np.load('perm-60000.npy')
# 100 label samples, 59900 unlabel samples
xla = torch.from_numpy( xtr[perm[0:100]] )
yla = torch.from_numpy( ytr[perm[0:100]] )
xun = torch.from_numpy( xtr[perm[100:]] )
yun = torch.from_numpy( ytr[perm[100:]] )
xte = torch.from_numpy( xte )
yte = torch.from_numpy( yte )

la = data.TensorDataset(xla, yla)
un = data.TensorDataset(xun, yun)
te = data.TensorDataset(xte, yte)
la_loader = data.DataLoader(la, batch_size=batchsize, shuffle=True)
un_loader = data.DataLoader(un, batch_size=batchsize, shuffle=True)
te_loader = data.DataLoader(te, batch_size=batchsize)

# Set torch evironment
torch.manual_seed(0)
device = torch.device("cuda")

# Build Model
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), 0.001)

# Train SL
for epoch in range(50):
    train(model, device, la_loader, optimizer, epoch)
    test(model, device, te_loader)

# Train SSL
print('Train SSL')
for epoch in range(10):
    train_ssl(model, device, la_loader, un_loader, optimizer, epoch)
    test(model, device, te_loader)
