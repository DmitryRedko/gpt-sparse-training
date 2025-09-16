import numpy as np
import torch


def _tensor_to_device(x: torch.Tensor, device: str='cuda'):
    if device.startswith('cuda'):
        return x.pin_memory().to(device, non_blocking=True)
    return x.to(device)

def prepare_random_loader(data: np.ndarray, batch_size: int, iters: int, block_size: int, device='cuda'):
    windows = torch.from_numpy(data).unfold(dimension=0, size=block_size, step=1)
    for _ in range(iters):
        ix = torch.randint(windows.size(0) - 1, (batch_size,))
        x, y = windows[ix], windows[ix + 1]
        yield _tensor_to_device(x, device), _tensor_to_device(y, device)

def prepare_sequential_loader(data: np.ndarray, batch_size: int, block_size: int, device='cuda'):
    windows = torch.from_numpy(data).view(-1, block_size)
    for i in range(0, windows.size(0), batch_size):
        x, y = windows[i : i + batch_size, :-1], windows[i : i + batch_size, 1:]
        yield _tensor_to_device(x, device), _tensor_to_device(y, device)
