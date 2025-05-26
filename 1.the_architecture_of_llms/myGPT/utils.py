import torch

def get_device():
    """Check available computing devices on Mac ARM"""
    print("PyTorch version:", torch.__version__)
    
    # Check if CUDA is available (unlikely on Mac)
    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
    else:
        print("CUDA is not available")
    
    # Check if MPS (Metal Performance Shaders) is available on Mac
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available")
        print("MPS device can be used for GPU acceleration on Mac")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    print(f"Selected device: {device}")
    return device

