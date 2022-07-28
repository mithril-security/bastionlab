import torch
from torch.nn import Module, Linear
from utils import remote_module
from private_module import PrivacyEngine

@remote_module
class LReg(Module):
    def __init__(self):
        super().__init__()
        self.w = Linear(1, 1, bias=False)

    def forward(self, x):
        return self.w(x)

engine = PrivacyEngine()

@remote_module
@engine.private_module(2)
class PrivateLReg(Module):
    def __init__(self):
        super().__init__()
        self.w = Linear(1, 1, bias=False)

    def forward(self, x):
        return self.w(x)

if __name__ == '__main__':
    print(engine.GLOBAL_SAMPLE_FN_TABLE)
    # script = torch.jit.script(LReg())
    # torch.jit.save(script, "lreg.pt")
    private_script = torch.jit.script(PrivateLReg())
    # private_script = PrivateLReg()
    # torch.jit.save(private_script, "private_lreg.pt")


    data = [
        torch.tensor([0.0, 1.0]).view(2, 1),
        torch.tensor([0.5, 0.2]).view(2, 1),
    ]

    target = [
        torch.tensor([0.0, 2.0]).view(2, 1),
        torch.tensor([1.0, 0.4]).view(2, 1),
    ]

    for x, t in zip(data, target):
        y = private_script(x)
        loss = (y - t).norm(2, dim=1).mean()
        loss.backward()
        print(list(private_script.grad_sample_parameters()))
        print([x[0].grad for x in private_script.grad_sample_parameters()])
        break
