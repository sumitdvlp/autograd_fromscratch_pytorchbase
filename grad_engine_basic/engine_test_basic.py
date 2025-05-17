import torch
import torch.nn as nn
import torch.nn.functional as F
import base 
import loss


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

a_Engine = base.Engine(data = a_torch) # 2x2
b_Engine = base.Engine(data = b_torch) # 2 x 1
c_Engine = base.Engine(data = c_torch)
y_Engine = base.Engine(data = y_pred)


d_Engine = a_Engine.__matmul__(b_Engine)
e_Engine = d_Engine.__add__(c_Engine) # 3 x 2 * 2 x 3 > 3 x 3
f_Engine = e_Engine.relu()
loss_Engine = loss.mse(y_Engine, f_Engine)
# f = e - y

# print('b shape',b_Engine.data.shape)
# print('b.data  ',b_Engine.data)
# print('d shape',d_Engine.data.shape)
# print('d.data  ',d_Engine.data)
# print('e shape',e_Engine.data.shape)
# print('e.data ',e_Engine.data)
# print('f shape',f_Engine.data.shape)
# print('f.data ',f_Engine.data)
# print('loss shape',loss_Engine.data.shape)
# print('loss data',loss_Engine.data)
loss_Engine.backward()
loss_Engine._op
# print('loss grad',loss_Engine.grad)
# print('f.grad', f_Engine.grad)
# print('e.grad', e_Engine.grad)
# print('d.grad', d_Engine.grad)
# print('c.grad', c_Engine.grad)
# print('a.grad', a_Engine.grad)
# print('b.grad', b_Engine.grad)



a_torch.requires_grad = True
b_torch.requires_grad = True
c_torch.requires_grad = True
y_pred.requires_grad = True


d_torch = torch.mm(a_torch,b_torch)
e_torch = torch.add(d_torch, c_torch) # 3 x 2 * 2 x 3 > 3 x 3
f_torch = F.relu(e_torch)
loss = loss.mse_pytorch(y_pred,f_torch)
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
     verify
'''
a_check = torch.ne(a_torch.grad, a_Engine.data.grad)
b_check = torch.ne(b_torch.grad, b_Engine.data.grad)
c_check = torch.ne(c_torch.grad, c_Engine.data.grad)
print(torch.where(a_check == True))
print(torch.where(b_check == True))
print(torch.where(c_check == True))
# torch.eq(d_torch.grad, d_Engine.data.grad)