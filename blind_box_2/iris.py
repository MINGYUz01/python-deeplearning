from unittest import TestLoader
from numpy import corrcoef
from sklearn.datasets import load_iris
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import swanlab

class Tools():
    def get_iris(self):
        iris_data = load_iris()
        data, target = iris_data['data'], iris_data['target']
        feature_names = iris_data['feature_names']
        target_names = iris_data['target_names']
        return data, target, feature_names, target_names
    
    def show_losses(self, train_losses, val_losses):
        epochs = list(range(1, len(train_losses)+1))
        plt.title('train/val loss')
        plt.plot(epochs, train_losses, c='g', label='train loss')
        min_loss = min(val_losses)
        min_idx = val_losses.index(min_loss)+1
        plt.plot(epochs, val_losses, c='b', label='val loss')
        plt.scatter(min_idx, min_loss, c='r', label='min val loss:{:.4f}'.format(min_loss))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.show()

    def show_valacc(self, val_acc):
        epochs = list(range(1, len(val_acc)+1))
        plt.title('val acc')
        plt.plot(epochs, val_acc, c='g', label='val acc')
        max_acc = max(val_acc)
        max_idx = val_acc.index(max_acc)+1
        plt.scatter(max_idx, max_acc, c='r', label='max acc:{:.4f}'.format(max_acc))
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.grid(True)
        plt.legend()
        plt.show()

class MyDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.long)
    def __len__(self):
        return self.target.size(0)
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softmax()
        )
    def forward(self, x):
        z = self.model(x)
        return z

def train(model, epochs, train_loader, val_loader, device, optimizer, criterion):
    train_losses = []
    val_losses = []
    val_acc = []
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        running_iter = 0
        model.train()
        for x, y in tqdm(train_loader, desc='Train Epoch {}/{}'.format(epoch, epochs)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            running_loss += loss.item() * y.size(0)
            running_iter += y.size(0)
            loss.backward()
            optimizer.step()
        train_loss = running_loss / running_iter
        swanlab.log({
            'train/loss': train_loss
        }, step=epoch)
        train_losses.append(train_loss)

        running_loss = 0.0
        running_iter = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for x, y, in tqdm(val_loader, desc='Val Epoch {}/{}'.format(epoch, epochs)):
                x, y = x.to(device), y.to(device)
                z = model(x)
                loss = criterion(z, y)
                running_loss += loss.item() * y.size(0)
                running_iter += y.size(0)
                predicts = torch.argmax(z, -1)
                correct += torch.sum(predicts == y).cpu().item()
        val_loss = running_loss / running_iter
        acc = correct / running_iter
        val_losses.append(val_loss)
        val_acc.append(acc)
        swanlab.log({
            'val/loss': val_loss,
            'val/acc': acc
        }, step=epoch)
                
    return train_losses, val_losses, val_acc

def test(model, test_loader):
    coreect = 0
    iterations = 0
    with torch.no_grad():
        model.eval()
        for x, y in tqdm(test_loader, 'Testing ...'):
            z = model(x)
            predict = torch.argmax(z, -1)
            coreect += torch.sum(predict == y).item()
            iterations += y.size(0)
    test_acc = coreect / iterations
    print('Test acc : {}'.format(test_acc))


def main():
    config = {
        'cwd': os.getcwd(),
        'model': 'linear(4, 3)',
        'dataset': 'iris',
        'epochs': 300,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'optimizer': 'Adam',
        'criterion': 'MSELoss',
    }
    run = swanlab.init(
        project='blind_box_2',
        experiment_name='iris',
        description='这是一个关于莺尾草分类的分类问题',
        config=config
    )
    tools = Tools()
    data, target, feature_names, target_names = tools.get_iris()
    dataset = MyDataset(data, target)
    run.config.set('dataset_size', len(dataset))
    train_rate, val_rate = 0.8, 0.15
    train_size, val_size = int(len(dataset)*train_rate), int(len(dataset)*val_rate)
    test_size = len(dataset) - train_size - val_size
    run.config.set('train_size', train_size)
    run.config.set('val_size', val_size)
    run.config.set('test_size', test_size)

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader, val_loader = DataLoader(train_set, run.config.batch_size, True), DataLoader(val_set, run.config.batch_size, True)
    test_loader = DataLoader(test_set, 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run.config.set('device', device)
    input_dim, output_dim = len(feature_names), len(target_names)
    run.config.set('feature_names', feature_names)
    run.config.set('target_names', target_names)
    my_model = MyModel(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=run.config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_acc = train(
        model=my_model,
        epochs=run.config.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        criterion=criterion
    )

    tools.show_losses(train_losses, val_losses)
    tools.show_valacc(val_acc)

if __name__ == '__main__':
    main()