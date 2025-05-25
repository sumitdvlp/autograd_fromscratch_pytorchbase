from autograd_engine import utils as engine
import torch
import unittest
from autograd_engine.nn_modules import layers
import torch.nn as nn

class TestNeuralForge(unittest.TestCase):
    ''' This class tests the functionalities of the framework in three levels of complexity. '''

    def test_autograd(self):
        '''
        This function tests whether the loss converges to zero in a spelled-out forward
        propagation, with weights explicitly declared.
        '''
        # Define loss function as Cross Entropy Loss:
        loss_func = layers.MSE()

        # Instantiate input and output:
        x = engine.randn((8,4,5))
        y = torch.randint(0,50,(8,4))

        # Instantiate Neural Network's Layers:
        w1 = engine.tensor(torch.randn(5,128) / torch.sqrt(torch.tensor(5)), requires_grad=True) 
        relu1 = layers.ReLU()
        w2 = engine.tensor(torch.randn(128,128) / torch.sqrt(torch.tensor(128)), requires_grad=True)
        relu2 = layers.ReLU()
        w3 = engine.tensor(torch.randn(128,50) / torch.sqrt(torch.tensor(128)), requires_grad=True)

        # Training Loop:
        for _ in range(4000):
            z = x @ w1
            z = relu1(z)
            z = z @ w2
            z = relu2(z)
            z = z @ w3
            
            # Get loss:
            loss = loss_func(z, y)

            # Backpropagate the loss using neuralforge.tensor:
            loss.backward()

            # Update the weights:
            w1 = w1 - (w1.grad * 0.005) 
            w2 = w2 - (w2.grad * 0.005) 
            w3 = w3 - (w3.grad * 0.005) 

            # Reset the gradients to zero after each training step:
            loss.zero_grad_tree()
        assert loss._data < 3e-1, "Error: Loss is not converging ttesto zero in autograd test."


seed_value=42
torch.manual_seed(seed_value)

if __name__ == '__main__':
    unittest.main()