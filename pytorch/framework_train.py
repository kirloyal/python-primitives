# ref)
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# https://github.com/pytorch/examples/blob/master/mnist/main.py
# https://pytorch.org/docs/stable/tensorboard.html

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
from framework_model import Net
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--load-model', type=str, default='', help='For Loading the saved Model')
parser.add_argument('--save-model', type=str, default='./tmp/mnist_cnn.pt', help='For Saving the current Model')
args = parser.parse_args()

torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
trainset = datasets.MNIST('./tmp', train=True, download=True, transform=transform)            
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = datasets.MNIST('./tmp', train=False, transform=transform)            
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = F.nll_loss

epoch_start = 0
best_test_loss = float("inf")
if args.load_model != '':
    state = torch.load(args.load_model)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    epoch_start = state['epoch']
    best_test_loss = state['loss']

writer_correct = SummaryWriter('./tmp/correct')
writer_train = SummaryWriter('./tmp/train_0')
writer_test = SummaryWriter('./tmp/test_0')


for epoch in range(epoch_start + 1, epoch_start + args.epochs + 1):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # reduction = 'mean' by default
        loss.backward()
        optimizer.step()
        train_loss += loss * len(data)
        if batch_idx % args.log_interval == 0:
            pbar.set_description("Train Epoch: {}, Loss: {:.6f}".format(epoch, loss.item()))
    train_loss /= len(train_loader.dataset)    
    
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
    
    writer_train.add_scalar('loss', train_loss, epoch)
    writer_test.add_scalar('loss', test_loss, epoch)
    writer_correct.add_scalar('correct', correct, epoch)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if test_loss < best_test_loss:
        print(epoch)
        best_test_loss = test_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss
            }, args.save_model)


writer_train.close()
writer_test.close()
writer_correct.close()
