#%%

import torch
import torch.nn as nn
from torchvision import datasets, transforms


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
dataset = datasets.CIFAR10('../datasets', train=False, download=False, transform=transform)
dataloader = torch.utils.data.DataLoader( dataset = dataset, batch_size=1, shuffle=True, num_workers=1)

X, y = next(iter(dataloader))
X = X.to(device)
y = y.to(device)

#%%

import torch
import torch.nn as nn
import torchvision.models as models

vgg16 = models.vgg16(pretrained=True).eval()
print(X.shape)
y_out = vgg16(X)
print(y_out.shape)

#%%

data = X
print(data.shape)
data = vgg16.features(data)
print(data.shape)
data = vgg16.avgpool(data)
print(data.shape)
data = data.view(data.size(0), -1)
print(data.shape)
data = vgg16.classifier(data)
print(data.shape)

print(y_out.equal(data))


#%%

data = X

vgg_feature_layers=['conv1_1','relu1_1','conv1_2','relu1_2','pool1','conv2_1',
                         'relu2_1','conv2_2','relu2_2','pool2','conv3_1','relu3_1',
                         'conv3_2','relu3_2','conv3_3','relu3_3','pool3','conv4_1',
                         'relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                         'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']

last_layer = 'pool4'
last_layer_idx = vgg_feature_layers.index(last_layer)
model = nn.Sequential(*list(vgg16.features.children())[:last_layer_idx+1])

y = model(X)
print(y.shape)


#%%
