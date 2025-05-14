import os
from sklearn.datasets import load_iris
import sklearn.datasets as datasets
from PIL import Image
from torch import nn

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
    
if __name__ == '__main__':
    model = MyModel()
    print(type(model.__str__()))