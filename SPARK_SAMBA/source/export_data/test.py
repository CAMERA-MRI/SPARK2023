import torch 

print("torch.cuda.is_available?", torch.cuda.is_available())
print("torch.cuda.current_device()?", torch.cuda.current_device())
print("torch.cuda.device(0)", torch.cuda.device(0))
print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))