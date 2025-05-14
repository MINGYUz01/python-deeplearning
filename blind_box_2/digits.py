from turtle import forward
import numpy as np
import os
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import load_digits
from tqdm import tqdm
import swanlab

def get_data():
    digits_data = load_digits()
    images, target = digits_data['images'], digits_data['target']
    target_names = digits_data['target_names']
    print(images.shape, target.shape)
    return images, target, target_names

class MyDataset(Dataset):
    def __init__(self, images, target):
        super().__init__()
        self.images = torch.tensor(images, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.long)
    def __len__(self):
        return self.target.size(0)
    def __getitem__(self, index):
        x = self.images[index]
        y = self.target[index]
        return x, y
    
class MyModel(nn.Module):
    def __init__(self, image_size=(8, 8), digit_nums=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=image_size[0]*image_size[1], out_features=100),
            nn.ReLU(),
            nn.Linear(100, digit_nums),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.model(x)
        return z
    
def train(model, epochs, train_loader, val_loader, optimizer, criterion, device):
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        running_iter = 0
        model.train()
        for x, y in tqdm(train_loader, desc='Train Epoch {}/{}'.format(epoch, epochs)):
            x, y = x.to(device), y.to(device)
            z = model(x)
            optimizer.zero_grad()
            loss = criterion(z, y)
            running_loss += loss.item() * y.size(0)
            running_iter += y.size(0)
            loss.backward()
            optimizer.step()
        train_loss = running_loss / running_iter
        swanlab.log({
            'train/loss': train_loss
        }, epoch, True)

        with torch.no_grad():
            running_loss = 0.0
            running_iter = 0
            correct = 0
            model.eval()
            for x, y in tqdm(val_loader, desc='Val Epoch {}/{}'.format(epoch, epochs)):
                x, y = x.to(device), y.to(device)
                z = model(x)
                loss = criterion(z, y)
                running_loss += loss.item() * y.size(0)
                running_iter += y.size(0)
                pred = torch.argmax(z, -1)
                correct += torch.sum(pred == y).item()
            val_loss = running_loss / running_iter
            val_acc = correct / running_iter
            swanlab.log({
                'val/loss': val_loss,
                'val/acc': val_acc
            }, epoch, True)
        # print('Epoch: {}, Train_loss: {:.6f}, Val_loss: {:.6f}, Val_acc: {:.6f}'.format(epoch, train_loss, val_loss, val_acc))

        

def main():
    config = {
        'cwd': os.getcwd(),
        'model': '',
        'dataset': 'sklearn.datasets.load_digits',
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 1e-2,
        'optimizer': 'Adam',
        'criterion': 'CrossEntropyLoss',
    }
    run = swanlab.init(
        project='blind_box_2',
        experiment_name='digits',
        description='一个简单的digits多分类问题',
        config=config
    )
    images, target, target_names = get_data()
    run.config.set('target_names', target_names)
    image_size = images[0].shape
    run.config.set('image_size', image_size)
    digit_nums = len(target_names)
    run.config.set('digit_nums', digit_nums)
    print(image_size, digit_nums)

    dataset = MyDataset(images, target)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    run.config.set('train_size', train_size)
    run.config.set('val_size', val_size)
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader, val_loader = DataLoader(train_set, run.config.batch_size, shuffle=True), DataLoader(val_set, run.config.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run.config.set('device', device)
    my_model = MyModel(image_size, digit_nums).to(device)
    run.config.set('model', my_model.__str__())
    optimizer = Adam(my_model.parameters(), lr=run.config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train(my_model, run.config.epochs, train_loader, val_loader, optimizer, criterion, device)

if __name__ == '__main__':
    main()