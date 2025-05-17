from autograd_engine import utils as engine
import torch
import unittest


class TestCoreOperations(unittest.TestCase):
    def test_add(self):
        at = torch.rand(2,1)
        bt = torch.rand(2,1)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data=bt, requires_grad=True,op='b')

        c = a + b
    


        ct = at + bt
        ct.requires_grad = True
        
        assert torch.equal(ct,c._data)," addition didn't worked"


        bc = c.sum()
        bct = ct.sum()
        bc.backward()
        bct.backward()

        assert torch.equal(c.grad, ct.grad), " addition backward pass didn't worked"

seed_value=42
torch.manual_seed(seed_value)

if __name__ == '__main__':
    unittest.main()