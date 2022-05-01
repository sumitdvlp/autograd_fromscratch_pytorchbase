def mse(y,y_pred):
  '''    
  squred_diff = (input_tensor - target_tensor) ** 2
      return torch.sum(squred_diff)/input_tensor.shape[0]
      '''
  diff = y - y_pred
  sq = diff ** 2
  return sq.sum()/y.shape()[0]

def mse_pytorch(y,y_pred):
  squred_diff = (y - y_pred) ** 2
  return torch.sum(squred_diff)/y.shape[0]