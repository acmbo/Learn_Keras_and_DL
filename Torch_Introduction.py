# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 14:51:11 2020

@author: Stephan
"""


#Website :https://nbviewer.jupyter.org/github/cgpotts/cs224u/blob/master/tutorial_pytorch.ipynb

import torch

print("Pytorch Version {}".format(torch.__version__))
print("GPU-enabled installation? {}".format(torch.cuda.is_available()))

'''Calling Constructor Torch'''

t = torch.FloatTensor(2, 3)
print(t)
print(t.size())

t.zero_() # Underscore means Operator acts inplace and constructs tensor


torch.FloatTensor([[1, 2, 3], [4, 5, 6]]) # Construcion from Array

'''Inbuild Functions '''

tl = torch.tensor([1, 2, 3])
t = torch.tensor([1., 2., 3.])
print("A 64-bit integer tensor: {}, {}".format(tl, tl.type()))
print("A 32-bit float tensor: {}, {}".format(t, t.type()))


t = torch.zeros(2, 3)
print(t)

t_zeros = torch.zeros_like(t)        # zeros_like returns a new tensor
t_ones = torch.ones(2, 3)            # creates a tensor with 1s
t_fives = torch.empty(2, 3).fill_(5) # creates a non-initialized tensor and fills it with 5
t_random = torch.rand(2, 3)          # creates a uniform random tensor
t_normal = torch.randn(2, 3)         # creates a normal random tensor

print(t_zeros)
print(t_ones)
print(t_fives)
print(t_random)
print(t_normal)




# creates a new copy of the tensor that is still linked to 
# the computational graph (see below)
t1 = torch.clone(t)
assert id(t) != id(t1), 'Functional methods create a new copy of the tensor'

# To create a new _independent_ copy, we do need to detach 
# from the graph
t1 = torch.clone(t).detach()


import numpy as np

# Create a new multi-dimensional array in NumPy with the np datatype (np.float32)
a = np.array([1., 2., 3.])

# Convert the array to a torch tensor
t = torch.tensor(a)

print("NumPy array: {}, type: {}".format(a, a.dtype))
print("Torch tensor: {}, type: {}".format(t, t.dtype))


t.numpy() # Convert tensor into numpy array


t = torch.randn(2, 3) 
t[ : , 0] #indexing




t = torch.randn(5, 6)
print(t)
i = torch.tensor([1, 3])
j = torch.tensor([4, 5])
print(t[i])                          # selects rows 1 and 3
print(t[i, j])                       # selects (1, 4) and (3, 5)

t = t.float()   # converts to 32-bit float
print(t)
t = t.double()  # converts to 64-bit float
print(t)
t = t.byte()    # converts to unsigned 8-bit integer
print(t)



# Scalars =: creates a tensor with a scalar 
# (zero-th order tensor,  i.e. just a number)
s = torch.tensor(42)
print(s)
s.item()



# Row vector
x = torch.randn(1,3)
print("Row vector\n{}\nwith size {}".format(x, x.size()))

# Column vector
v = torch.randn(3,1)
print("Column vector\n{}\nwith size {}".format(v, v.size()))

# Matrix
A = torch.randn(3, 3)
print("Matrix\n{}\nwith size {}".format(A, A.size()))


s = torch.matmul(x, torch.matmul(A, v))
print(s.item())

# common tensor methods (they also have the counterpart in 
# the torch package, e.g. as torch.sum(t))
t = torch.randn(2,3)
t.sum(dim=0)                 
t.t()                   # transpose
t.numel()               # number of elements in tensor
t.nonzero()             # indices of non-zero elements
t.view(-1, 2)           # reorganizes the tensor to these dimensions
t.squeeze()             # removes size 1 dimensions
t.unsqueeze(0)          # inserts a dimension

# operations in the package
torch.arange(0, 10)     # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
torch.eye(3, 3)         # creates a 3x3 matrix with 1s in the diagonal (identity in this case)
t = torch.arange(0, 3)
torch.cat((t, t))       # tensor([0, 1, 2, 0, 1, 2])
torch.stack((t, t))     # tensor([[0, 1, 2],
                        #         [0, 1, 2]])



t_gpu = torch.cuda.FloatTensor(3, 3)   # creation of a GPU tensor
t_gpu.zero_()                          # initialization to zero


'''GPU computation'''


t_gpu = torch.cuda.FloatTensor(3, 3)   # creation of a GPU tensor
t_gpu.zero_()                          # initialization to zero

try:
    t_gpu = torch.randn(3, 3, device="cuda:0")
except:
    print("Torch not compiled with CUDA enabled")
    t_gpu = None
    
# we could also state explicitly the device to be the 
# CPU with torch.randn(3,3,device="cpu")
t = torch.randn(3, 3)   


t_gpu = t.to("cuda:0")  # copies the tensor from CPU to GPU
# note that if we do now t_to_gpu.to("cuda:0") it will 
# return the same tensor without doing anything else 
# as this tensor already resides on the GPU
print(t_gpu)
print(t_gpu.device)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)


# moves t to the device (this code will **not** fail if the 
# local machine has not access to a GPU)
t.to(device)




'''NEURONAL NETS'''


import torch.nn as nn



class MyCustomModule(nn.Module):
    '''
    Forwardeed with y = U(f(W(x))) , where f is ReLu
    '''
    def __init__(self, n_inputs, n_hidden, n_output_classes):
        # call super to initialize the class above in the hierarchy
        super(MyCustomModule, self).__init__()
        # first affine transformation
        self.W = nn.Linear(n_inputs, n_hidden)        
        # non-linearity (here it is also a layer!)
        self.f = nn.ReLU()
        # final affine transformation
        self.U = nn.Linear(n_hidden, n_output_classes) 
        
    def forward(self, x):
        y = self.U(self.f(self.W(x)))
        return y




# set the network's architectural parameters
n_inputs = 3
n_hidden= 4
n_output_classes = 2

# instantiate the model
model = MyCustomModule(n_inputs, n_hidden, n_output_classes)

# create a simple input tensor 
# size is [1,3]: a mini-batch of one example, 
# this example having dimension 3
x = torch.FloatTensor([[0.3, 0.8, -0.4]]) 

# compute the model output by **applying** the input to the module
y = model(x)

# inspect the output
print(y)



'''As a sequential Class'''


class MyCustomModule(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_output_classes):
        super(MyCustomModule, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output_classes))
        
    def forward(self, x):
        y = self.network(x)
        return y


#Better Readability
class MyCustomModule(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_output_classes):
        super(MyCustomModule, self).__init__()
        self.p_keep = 0.7
        self.network = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 2*n_hidden),
            nn.ReLU(),
            nn.Linear(2*n_hidden, n_output_classes),   
            # dropout argument is probability of dropping
            nn.Dropout(1 - self.p_keep),
            # applies softmax in the data dimension
            nn.Softmax(dim=1)                  
        )
        
    def forward(self, x):
        y = self.network(x)
        return y



'''
Another important package in torch.nn is Functional, 
typically imported as F. Functional contains many useful functions,
 from non-linear activations to convolutional, dropout, and even 
 distance functions. Many of these functions have counterpart
 implementations as layers in the nn package so that they can be
 easily used in pipelines like the one above implemented using 
 nn.Sequential.
'''


import torch.nn.functional as F

y = F.relu(torch.FloatTensor([[-5, -1, 0, 5]]))



# the true label (in this case, 2) from our dataset wrapped 
# as a tensor of minibatch size of 1
y_gold = torch.tensor([1])        
                                  
# our simple classification criterion for this simple example    
criterion = nn.CrossEntropyLoss() 

# forward pass of our model (remember, using apply instead of forward)
y = model(x)  

# apply the criterion to get the loss corresponding to the pair (x, y)
# with respect to the real y (y_gold)
loss = criterion(y, y_gold)       
                                 

# the loss contains a gradient function that we can use to compute
# the gradient dL/dw (gradient with respect to the parameters 
# for a given fixed input)
print(loss)                                                     




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import math

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



M = 1200

# sample from the x axis M points
x = np.random.rand(M) * 2*math.pi

# add noise
eta = np.random.rand(M) * 0.01

# compute the function
y = np.sin(x) + eta

# plot
_ = plt.scatter(x,y)



# use the NumPy-PyTorch bridge
x_train = torch.tensor(x[0:1000]).float().view(-1, 1).to(device)
y_train = torch.tensor(y[0:1000]).float().view(-1, 1).to(device)

x_test = torch.tensor(x[1000:]).float().view(-1, 1).to(device)
y_test = torch.tensor(y[1000:]).float().view(-1, 1).to(device)



class SineDataset(data.Dataset):
    def __init__(self, x, y):
        super(SineDataset, self).__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

sine_dataset = SineDataset(x_train, y_train)

sine_dataset_test = SineDataset(x_test, y_test)

sine_loader = torch.utils.data.DataLoader(
    sine_dataset, batch_size=32, shuffle=True)

sine_loader_test = torch.utils.data.DataLoader(
    sine_dataset_test, batch_size=32)



class SineModel(nn.Module):
    def __init__(self):
        super(SineModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1))
        
    def forward(self, x):
        return self.network(x)

# declare the model
model = SineModel().to(device)

# define the criterion
criterion = nn.MSELoss()

# select the optimizer and pass to it the parameters of the model it will optimize
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

epochs = 1000

# training loop
for epoch in range(epochs):
    for i, (x_i, y_i) in enumerate(sine_loader):

        y_hat_i = model(x_i)            # forward pass
                                
        loss = criterion(y_hat_i, y_i)  # compute the loss and perform the backward pass

        optimizer.zero_grad()           # cleans the gradients
        loss.backward()                 # computes the gradients
        optimizer.step()                # update the parameters

    if epoch % 20:
        plt.scatter(x_i.data.cpu().numpy(), y_hat_i.data.cpu().numpy())
        
        

# testing
with torch.no_grad():
    model.eval()
    total_loss = 0.
    for k, (x_k, y_k) in enumerate(sine_loader_test):
        y_hat_k = model(x_k)
        loss_test = criterion(y_hat_k, y_k)
        total_loss += float(loss_test)

print(total_loss)



def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    # For atomic operations there is currently 
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    #
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    np.random.seed(seed)

enforce_reproducibility()


