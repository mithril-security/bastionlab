import torch
from torch.nn import Module
from psg.nn import Linear
from utils import remote_module

@remote_module
class PrivateLReg(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(1, 1, 2, bias=False)

    def forward(self, x):
        return self.fc1(x)

if __name__ == '__main__':
    private_script = torch.jit.script(PrivateLReg())
    torch.jit.save(private_script, "tests/private_lreg.pt")
