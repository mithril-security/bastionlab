import inspect
import torch
from typing import Callable, Iterable, Tuple, List, Dict
from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, Module
from torch.nn.functional import relu
from torch.nn.parameter import Parameter

def tensor_arity(fn: Callable) -> int:
    arity = 0
    signature = inspect.signature(fn)
    for param in signature.parameters.values():
        if param.annotation == Tensor:
            arity += 1
    return arity

# def create_or_accumulate_grad_sample(store: Dict[Parameter, Tensor], param: Parameter, grad_sample: Tensor, max_batch_len: int) -> None:
#     if param.requires_grad:
#         if param in store and store[param].size() != (0,):
#             store[param][: grad_sample.shape[0]] += grad_sample
#         else:
#             store[param] = Parameter(torch.zeros(
#                 torch.Size([max_batch_len]) + grad_sample.shape[1:],
#                 device=grad_sample.device,
#                 dtype=grad_sample.dtype,
#             ), requires_grad=False)
#             store[param][: grad_sample.shape[0]] = grad_sample

def parametrized_modules(module: Module) -> Iterable[Module]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters (as opposed to "wrapper modules" that just organize modules).
    """
    yield from (
        (m_name, m) # type: ignore
        for (m_name, m) in module.named_modules()
        if any(p is not None for p in m.parameters(recurse=False))
    )

def trainable_modules(module: Module) -> Iterable[Tuple[str, Module]]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters and are trainable (ie they want a grad).
    """
    yield from (
        (m_name, m)
        for (m_name, m) in parametrized_modules(module) # type: ignore
        if any(p.requires_grad for p in m.parameters(recurse=False))
    )


def trainable_parameters(module: Module) -> Iterable[Tuple[str, Parameter]]:
    """
    Recursively iterates over all parameters, returning those that
    are trainable (ie they want a grad).
    """
    yield from (
        (p_name, p) for (p_name, p) in module.named_parameters() if p.requires_grad
    )

class PrivacyEngine():
    GLOBAL_SAMPLE_FN_TABLE = {}

    def __init__(self) -> None:
        self.grad_sample_fn_table = {}

    def grad_sample_fn(self, cls: Callable) -> Callable[[Callable], None]:
        def inner(fn: Callable) -> None:
            self.grad_sample_fn_table[cls] = fn
        return inner

    def initialize_grad_sample_store(self, module: Module, max_batch_len: int):
        for _, param in trainable_parameters(module):
            if not hasattr(module, "grad_sample_store"):
                module.grad_sample_store = {}
            module.grad_sample_store[param] = Parameter(torch.zeros( # type: ignore
                torch.Size([max_batch_len, *param.size()]),
                device=param.device,
                dtype=param.dtype,
            ), requires_grad=False)

    def register_sample_hooks(self, module: Module):
        module.arity = tensor_arity(module.forward)
        module.register_forward_hook(self.sample_forward_hook(module.arity))
        module.register_full_backward_hook(self.sample_backprop_hook()) # type: ignore
        module.max_batch_len = 0
        module.test = { "test": Parameter(torch.ones(3)) }
        if module.arity == 1:
            empty1: List[Tuple[Tensor]] = [(torch.zeros(0),)]
            module.input = empty1 
        elif module.arity == 2:
            empty2: List[Tuple[Tensor, Tensor]] = [(torch.zeros(0), torch.zeros(0))]
            module.input = empty2

    def private_module(self, max_batch_len: int) -> Callable[[Callable], Callable]:
        def inner(cls: Callable) -> Callable:
            init = cls.__init__
            def new_init(_self, *args, **kwargs):
                init(_self, *args, **kwargs) # type: ignore
                self.make_private(_self, max_batch_len)
            cls.__init__ = new_init

            @torch.jit.export # type: ignore
            def grad_sample_parameters(_self) -> List[Tuple[Parameter, Parameter]]:
                return _self._grad_sample_parameters
            cls.grad_sample_parameters = grad_sample_parameters

            # @torch.jit.export
            # def get_grad_sample(_self, name: str) -> Parameter:
            #     path = name.split(".")
            #     target = _self
            #     for segment in path[:-1]:
            #         target = getattr(target, segment)
            #     param = getattr(target, path[-1])
            #     grad_sample = target.grad_sample_store[param]
            #     return grad_sample
            # cls.get_grad_sample = get_grad_sample

            return cls
        return inner

    def make_private(self, module: Module, max_batch_len: int) -> None:
        module._grad_sample_parameters = []
        for _, m in trainable_modules(module):
            if not type(m) in self.grad_sample_fn_table and not type(m) in PrivacyEngine.GLOBAL_SAMPLE_FN_TABLE:
                continue
            self.register_sample_hooks(m)
            self.initialize_grad_sample_store(m, max_batch_len)
            for _, p in trainable_parameters(m):
                module._grad_sample_parameters.append((p, m.grad_sample_store[p])) # type: ignore
        # module._grad_sample_parameters = [[m.grad_sample_store[p] for _, p in trainable_parameters(m)] for _, m in trainable_modules(module) if not type(m) in self.grad_sample_fn_table and not type(m) in PrivacyEngine.GLOBAL_SAMPLE_FN_TABLE]

        # for _, p in trainable_parameters(module):
        #     p.grad_sample = None # type: ignore

    def sample_forward_hook(self, arity: int) -> Callable[[Module, Tuple[Tensor, ...], Tensor], None]: # type: ignore
        if arity == 1:
            def inner1(module: Module, input: Tuple[Tensor], output: Tensor):
                module.input.append((input[0].detach(),)) # type: ignore
                if not module.max_batch_len > 0: # type: ignore
                    module.max_batch_len = input[0].shape[0]
            return inner1
        elif arity == 2:
            def inner2(module: Module, input: Tuple[Tensor, Tensor], output: Tensor):
                module.input.append((input[0].detach(), input[1].detach())) # type: ignore
                if not module.max_batch_len > 0: # type: ignore
                    module.max_batch_len = input[0].shape[0]
            return inner2
        else:
            raise Exception("Arity above two not yet supported")

    def sample_backprop_hook(self) -> Callable[[Module, Tuple[Tensor, ...], Tuple[Tensor, ...]], None]:
        # This compiles to torchscript without annotations so let's avoid
        # making another if/else on arity...
        def inner(module, grad_input, grad_output):
            if not type(module) in self.grad_sample_fn_table:
                if not type(module) in PrivacyEngine.GLOBAL_SAMPLE_FN_TABLE:
                    return
                grad_sample_fn = PrivacyEngine.GLOBAL_SAMPLE_FN_TABLE[type(module)]
            else:
                grad_sample_fn = self.grad_sample_fn_table[type(module)]

            for param, grad_sample in grad_sample_fn(module, module.input.pop(), grad_output).items():
                if param in module.grad_sample_store:
                    module.grad_sample_store[param][: grad_sample.shape[0]] += grad_sample
        return inner

def global_grad_sample_fn(cls: Callable) -> Callable[[Callable], None]:
    def inner(fn: Callable) -> None:
        PrivacyEngine.GLOBAL_SAMPLE_FN_TABLE[cls] = fn
    return inner


@global_grad_sample_fn(Linear)
def linear_grad_sample_fn(layer: Module, input: Tuple[Tensor], grad_output: Tuple[Tensor]) -> Dict[Parameter, Tensor]:
    return { # type: ignore
        layer.weight: torch.einsum("n...i,n...j->nij", grad_output[0], input[0]),
        layer.bias: torch.einsum("n...k->nk", grad_output[0]),
    }

if __name__ == '__main__':
    engine = PrivacyEngine()

    @engine.private_module(max_batch_len=64)
    class FCModel(Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = Linear(784, 1568)
            self.fc2 = Linear(1568, 10)
            self.test = { "test": Parameter(torch.ones(3)) }

        def forward(self, x: Tensor) -> Tensor:
            x = x.view(-1, 784)
            x = self.fc1(x)
            x = relu(x)
            x = self.fc2(x)
            return x


    model = FCModel()
    scripted = torch.jit.script(model) # type: ignore
    input = torch.ones(10, 1, 28, 28)
    output = model(input)
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(output, torch.zeros(10, dtype=torch.int64))
    loss.backward()
    torch.jit.save(scripted, "scripted_model.pt") # type: ignore
    print(model.fc1.weight.grad.size()) # type: ignore
    print(model.fc1.grad_sample_store[model.fc1.weight].size()) # type: ignore
    print(scripted.fc1.grad_sample_store[model.fc1.weight].size()) # type: ignore
    print([(x.size(), y.size()) for x, y in scripted.grad_sample_parameters()]) # type: ignore

