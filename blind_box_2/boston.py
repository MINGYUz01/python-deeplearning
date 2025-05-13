from cProfile import label
import os
from cv2 import transform
from flask import config
from itsdangerous import want_bytes
import pandas as pd
import torch
from torch import nn as nn, orgqr
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import swanlab

class Tools():
    def get_data(self, root):
        file_path = os.path.join(root, 'datasets', 'boston', 'boston.csv')
        assert os.path.exists(file_path), "'{}' not found!".format(file_path)
        df = pd.read_csv(file_path, header=0).to_dict(orient='list')
        features = list(df.keys())[:-1]
        target = list(df.keys())[-1]
        return features, target, df
    
    def show_origin_predict(self, origin, predict):
        x = list(range(1, len(predict)+1))
        plt.title('origin predict')
        plt.plot(x, origin, c='g', ls='-.', label='origin')
        plt.plot(x, predict, c='r', ls='--', label='predict')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        swanlab.log({'origin&predict data': swanlab.Image(plt)})
        plt.show()
    
    def show_train_losses(self, train_losses):
        epochs = list(range(1, len(train_losses)+1))
        plt.title('train loss')
        min_loss = min(train_losses)
        min_idx = train_losses.index(min_loss)
        plt.plot(epochs, train_losses, c='b', label='loss')
        plt.scatter(min_idx, min_loss, c='r', label='min loss : {:.3f}'.format(min_loss))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.show()
    
class MyDataset(Dataset):
    def __init__(self, features, target, data):
        super().__init__()
        self.x = torch.tensor(np.array([data[k] for k in features]).T, dtype=torch.float32)
        self.y = torch.tensor(data[target], dtype=torch.float32).reshape(-1, 1)
        # self.x_mean = self.x.mean()
        # self.x_std = self.x.std() + 1e-8
        # self.x = (self.x - self.x_mean) / self.x_std  # 标准归一化

        # self.y_mean = self.y.mean()
        # self.y_std = self.y.std() + 1e-8
        # self.y = (self.y - self.y_mean) / self.y_std



    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.y)

class MyModel(nn.Module):
    def __init__(self, features_num, target_num):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(features_num, 20),
            nn.ReLU(),
            nn.Linear(20, target_num)
            # nn.Linear(features_num, target_num)
        )
    def forward(self, x):
        z = self.fc(x)
        return z
    
def train(model, epochs, train_loader, optimizer, criterion, device):
    train_losses = []
    best_loss = np.inf
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        running_iter = 0
        model.train()
        for x, y in tqdm(train_loader, desc='Train Epoch: {} / {}'.format(epoch, epochs)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            running_loss += loss.item() * y.size(0)
            running_iter += y.size(0)
            loss.backward()
            optimizer.step()
        train_loss = running_loss / running_iter
        train_losses.append(train_loss)
        swanlab.log({
            'train_loss': train_loss
        }, step=epoch)
    return train_losses

        

def main():

    config = {
        'cwd': os.getcwd(),
        'model': 'linear(13, 1)',
        'dataset': 'iris',
        'batch_size': 32,
        'learning_rate': 1e-2,
        'optimizer': 'Adam',
        'criterion': 'MSELoss',
        'epochs': 100
    }
    run = swanlab.init(
        project="blind_box_2",
        experiment_name="boston",
        description="关于波士顿房价预测的一个线性回归模型",
        config=config,
    )
    
    tools = Tools()
    features, target, data = tools.get_data(run.config.cwd)

    train_set = MyDataset(features, target, data)
    # print('len(train_set): {}, train_set[0]: {}'.format(len(train_set), train_set[0]))

    train_loader = DataLoader(train_set, run.config.batch_size, shuffle=True, drop_last=False)

    features_num, target_num = len(features), len(target) if isinstance(target, list) else 1
    run.config.set('features_num', features_num)
    run.config.set('target_num', target_num)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run.config.set('device', device)

    linear = MyModel(features_num, target_num).to(device)
    print(linear)

    optimizer = torch.optim.Adam(linear.parameters(), lr=run.config.learning_rate)
    criterion = nn.MSELoss()

    train_losses = train(
        model=linear,
        epochs=run.config.epochs,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    tools.show_train_losses(train_losses=train_losses)
    x = len(train_set)
    origin = data[target]
    linear.cpu()
    linear.eval()
    inputs = torch.tensor([data[k] for k in features]).T
    predict = linear(inputs).detach().numpy()
    # predict = predict * train_set.y_std.numpy() + train_set.y_mean.numpy()
    tools.show_origin_predict(origin=origin, predict=predict)



if __name__ == '__main__':
    main()