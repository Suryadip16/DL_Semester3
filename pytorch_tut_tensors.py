import torch
import numpy as np

# tensors from data

data = [[1, 2], [3, 4]]
x_tensor = torch.tensor(data)

# tensors from numpy arrays

np_arr = np.array(data)
x_tensor_from_np = torch.from_numpy(np_arr)


# types

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)


# tensor from another tensor

rand_tensor_like = torch.rand_like(x_tensor, dtype=torch.float)
ones_tensor_like = torch.ones_like(x_tensor_from_np)


# Tensor Attributes
# print(f"Shape: {rand_tensor.shape}")
# print(f"Data Type: {rand_tensor.dtype}")
# print(f"Device: {rand_tensor.device}")

# numpy lke indexing

# print(rand_tensor[0])
# print(rand_tensor[:, 0])

rand_tensor[:, 1] = 0
#print(rand_tensor)

# concatenation

t1 = torch.ones(2, 2)
# print(t1)

t1_concat_r = torch.cat([t1, t1, t1], dim=0)
# print(t1_concat_r)
# print(t1_concat_r.shape)

t1_concat_c = torch.cat([t1, t1, t1], dim=1)
# print(t1_concat_c)
# print(t1_concat_c.shape)

# Math

tensor = torch.ones(4, 4)
# 3 ways to perform matrix operations. @ performs matrix multiplication.
# .T does a transpose operation.

m1 = tensor @ tensor.T
m2 = tensor.matmul(tensor.T)
m3 = torch.rand_like(m1)
torch.matmul(tensor, tensor.T, out=m3)

# m1, m2, m3 all of them will be the same.
# print(m1)
# print(m2)
# print(m3)

# * performs element wise product

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(z1)
torch.mul(tensor, tensor, out=z3)

# z1, z2, z3 will all have the same result

# print(z1)
# print(z2)
# print(z3)

# Single-element tensors If you have a one-element tensor, for example by aggregating
# all values of a tensor into one value, you can convert it to a Python numerical
# value using item()

agg = tensor.sum()
agg_item = agg.item()
# print(agg_item, type(agg_item))

# In-place operations: They are denoted by a _ suffix.
# For example: x.copy_(y), x.t_(), will change x.

# print(tensor)
tensor.add_(5)
# print(tensor)






