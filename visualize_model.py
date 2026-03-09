# I was not able run this file on VSCode because of the 
# 'graphviz.backend.execute.ExecutableNotFound' error.
# I ran the code on Google Colab to get the model visualization.
# I will look into the error and fix it for the upcoming assignments.

import torch
import torch.nn as nn
from torchviz import make_dot

model = nn.Sequential(
    nn.Linear(784,512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(512,256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(256,128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(128,10)
)

model.eval()

x = torch.randn(2,784)
y = model(x)

make_dot(y, params=dict(model.named_parameters())).render("nn_architecture", format="pdf")
