from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from itertools import chain 

# class Sequence(nn.Module):
#     def __init__(self):
#         super(Sequence,self).__init__()
#         self.lstm1 = nn.GRU(1,64,1)
#         self.lstm2 = nn.GRU(64,1,1)
#         self.p = 0.5
        
#     def forward(self,seq, hc = None):
#         out = []
#         if hc == None:
#             hc1, hc2 = None, None
#         else:
#             hc1, hc2 = hc
#         X_in = torch.unsqueeze(seq[0],0)
#         for X in seq.chunk(seq.size(0),dim=0):
#             if np.random.rand()>self.p:
#                 X_in = X
#             tmp, hc1 = self.lstm1(X_in,hc1)
#             X_in, hc2 = self.lstm2(tmp,hc2)
#             out.append(X_in)
#         return torch.stack(out).squeeze(1),(hc1,hc2)

class backbone_GRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=1):
        super(backbone_GRU,self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        
    def forward(self, x_in, h_in = None):
        """
        param x : (seq_len, batch, input_size)
        """
        x_out, h_out = self.gru(x_in, h_in)
        return x_out, h_out

        
def batch(l = 100, b = 64):
    ret = np.sin(np.linspace(0,10, l)[:,None]+2*np.pi*np.random.rand(b)[None,:]).astype(np.float32)
    return ret

model = backbone_GRU()                
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for i in range(1000):
    data = batch(21)
    X = torch.tensor(data[:-1, ..., None])
    Y = torch.tensor(data[1:, ..., None])
    optimizer.zero_grad()
    out, h_out = model(X)
    print(X.shape, Y.shape, out.shape, h_out.shape)
    a=bbb
    loss = criterion(out[20:],Y[20:])
    loss.backward()
    optimizer.step()
    if i%10 == 0:
        print("i {}, loss {}".format(i,loss.data.numpy()))  


# y = np.sin(np.linspace(0,100,1000)+2*np.pi*np.random.rand())
# X = Variable(torch.Tensor(y))[:100].view(-1,1,1)
# lstm_out,hc = seq(X)
# preds = []
# pred = lstm_out[-1].view(1,1,1)
# for i in range(1000):
#     pred,hc = seq(pred,hc)
#     preds.append(pred.data.numpy().ravel()[0])

# plt.figure(figsize=[10,8])
# xs = np.arange(lstm_out.size(0))
# plt.plot(xs,lstm_out.data.numpy().ravel())
# xs = np.arange(1000)+len(xs)
# plt.plot(xs,preds)
# plt.plot(y)
# plt.legend(['seeding','GRU','real'])
# plt.title('GRU')