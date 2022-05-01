import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    #### USING AUTOGRAD ENGINE ######
'''
# a = torch.tensor([[1,2],[5,6]],dtype=float,requires_grad=True) # 2x2
# c = torch.tensor([[3],[1]],dtype=float,requires_grad=True) # 2 x 1
# b = torch.tensor([[2,1],[9,3]], dtype=float,requires_grad=True)
# y = torch.tensor([[4],[9]], dtype=float)
a_torch = torch.rand(4555,300,dtype=float)
b_torch = torch.rand(300,1,dtype=float)
c_torch = torch.rand(4555,1,dtype=float)
y_pred = torch.rand(4555,1,dtype=float)
# d = torch.mm(a,b)
# d = torch.add(d,c)



a_value = Value(a_torch) # 2x2
b_value = Value(b_torch) # 2 x 1
c_value = Value(c_torch)
y_value = Value(y_pred)
d_value = a_value.matmul(b_value)
e_value = d_value.__add__(c_value) # 3 x 2 * 2 x 3 > 3 x 3
f_value = e_value.relu()
loss_value = mse(y_value, f_value)
# f = e - y

# print('b shape',b_value.data.shape)
# print('b.data  ',b_value.data)
# print('d shape',d_value.data.shape)
# print('d.data  ',d_value.data)
# print('e shape',e_value.data.shape)
# print('e.data ',e_value.data)
# print('f shape',f_value.data.shape)
# print('f.data ',f_value.data)
# print('loss shape',loss_value.data.shape)
# print('loss data',loss_value.data)
loss_value.backward()
loss_value._op
# print('loss grad',loss_value.grad)
# print('f.grad', f_value.grad)
# print('e.grad', e_value.grad)
# print('d.grad', d_value.grad)
# print('c.grad', c_value.grad)
# print('a.grad', a_value.grad)
# print('b.grad', b_value.grad)



'''
    ### Using Pytorch
'''



a_torch.requires_grad = True
b_torch.requires_grad = True
c_torch.requires_grad = True
y_pred.requires_grad = True


d_torch = torch.mm(a_torch,b_torch)
e_torch = torch.add(d_torch, c_torch) # 3 x 2 * 2 x 3 > 3 x 3
f_torch = F.relu(e_torch)
loss = mse_pytorch(y_pred,f_torch)
# f = relu_layer(e)
# print('b shape',b_torch.data.shape)
# print('b_torch.data  ',b_torch.data)
# print('d shape',d_torch.data.shape)
# print('d.data  ',d_torch.data)
# print('e shape',e_torch.data.shape)
# print('e.data ',e_torch.data)
# print('f shape',f_torch.data.shape)
# print('f.data ',f_torch.data)
# print('loss shape',loss.data.shape)
# print('loss data',loss.data)
d_torch.retain_grad()
e_torch.retain_grad()
f_torch.retain_grad()
loss.retain_grad()
loss.backward(torch.ones_like(loss))
# print('loss grad',loss.grad)
# print('f.grad', f_torch.grad)
# print('e.grad', e_torch.grad)
# print('d.grad', d_torch.grad)
# print('c_torch.grad', c_torch.grad)
# print('a_torch.grad', a_torch.grad)
# print('b_torch.grad', b_torch.grad)



''' 
    Verify
'''

a_check = torch.ne(a_torch.grad, a_value.data.grad)
b_check = torch.ne(b_torch.grad, b_value.data.grad)
c_check = torch.ne(c_torch.grad, c_value.data.grad)
print(torch.where(a_check == True))
print(torch.where(b_check == True))
print(torch.where(c_check == True))
# torch.eq(d_torch.grad, d_value.data.grad)