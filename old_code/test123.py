import torch, sys
print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Python path:", sys.executable)
