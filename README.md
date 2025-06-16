# Autograd in PyTorch

## This is a re-implementation of PyTorch's autograd (`torch.autograd`). 

As you know, Pytorch contains 3 major components:
+ `tensor` can be seen as a replacement of `numpy` for both GPU and CPU because it has unified API.
+ `autograd.Variable` is an automatic differentiation tool given a forward formulation.
+ `nn`: is a deep learning framework build based on `tensor` and `autograd`.

This project is aimed to re-implement the `autograd` part because: 
+ Pytorch's autograd functions are mostly implemented in `C/C++` (for performance purposes) so it is much harder for create new autograd's function.
+ Instead of building the backward function of complex function, we can build just the forward function, the backward function will be automatically built based on autograd.
+ Understand clearly how autograd works is very important for serious deep learning learner. 
+ You don't need to build the complex computational graph to do back-propagation in deep learning!

The reasons why we choose pytorch tensor over numpy array:
+ Pytorch tensor supports GPU.
+ We can easily validate your own autograd function with Pytorch's autograd with the same API.
+ Getting familiar ourselves with Pytorch's API is a big plus for later deep learning project.

##  Features of autograd_fromscratch_pytorchbase
- Autograd engine build from scratch using pytorch tensor as data holder (require_grad not used instead backpropogation is manually calculated.)
- Its unit tested framework to understand how backpropgation works internally in simplar ways.
- The Core [Engine](https://github.com/sumitdvlp/autograd_fromscratch_pytorchbase/blob/141ff1c723c08a7c69e1c5b2792c7da411003c24/grad_engine/core/base.py#L4)
- The Core Tensor [Tensor](https://github.com/sumitdvlp/autograd_fromscratch_pytorchbase/blob/main/autograd_engine/core.py#L4)
