#%%

import torch
import torch.nn as nn
from torchvision import datasets, transforms


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
dataset = datasets.CIFAR10('../datasets', train=False, download=False, transform=transform)
dataloader = torch.utils.data.DataLoader( dataset = dataset, batch_size=4, shuffle=True, num_workers=1)

X, y = next(iter(dataloader))
X = X.to(device)
y = y.to(device)

#%%

import torch
import torch.nn as nn
import torchvision.models as models

resnet101 = models.resnet101(pretrained=True)
resnet_feature_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
last_layer = 'layer3'
last_layer_idx = resnet_feature_layers.index(last_layer)

resnet_module_list = [resnet101.conv1, resnet101.bn1, resnet101.relu, resnet101.maxpool, resnet101.layer1,
                    resnet101.layer2, resnet101.layer3, resnet101.layer4, resnet101.avgpool, resnet101.fc]

model = nn.Sequential(*resnet_module_list[:last_layer_idx+1])
print(X.shape)
y_out = model(X)
print(y_out.shape)
#%%

print(resnet101)
print()
for var in vars(resnet101):
    print(var)
print()
for item in resnet101._modules:
    print(item)
print(resnet101.fc)
print()
print(resnet101._modules['fc'])
print()

#%%

data = X

for i in range(last_layer_idx+1):
    print(data.shape)
    print(resnet_module_list[i])
    data = resnet_module_list[i](data)
    print()
print(data.shape)

#%%

data = model(X)
data = data.transpose(1,2).transpose(2,3).contiguous()
print(data.shape)
n = data.shape[0]*data.shape[1]*data.shape[2]
f = data.shape[3]
data = data.view(n, f)
print(data.shape)

#%%

print(X.shape)
print(resnet101(X).shape)

#%%

