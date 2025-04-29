import torch
print("CUDA disponible :", torch.cuda.is_available())
print("Version CUDA (compilée avec) :", torch.version.cuda)
print("Version cuDNN :", torch.backends.cudnn.version())
print("Nom du GPU :", torch.cuda.get_device_name(0))