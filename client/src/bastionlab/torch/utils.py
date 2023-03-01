from typing import List, Tuple, Optional
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

__pdoc__ = {}


class TensorDataset(Dataset):
    """A simple dataset compliant with Torch's `Dataset` build upon
    tensors representing columns and labels.

    Args:
        columns: Tensors that represent the clolumns of the dataset (a column contains the values
        for a given input for all samples).
        labels: A tensor containing the labels of all inputs.
    """

    def __init__(self, columns: List[Tensor], labels: Optional[Tensor]) -> None:
        super().__init__()
        self.columns = columns
        self.labels = labels

    def __len__(self) -> int:
        return len(self.columns[0])

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], Optional[Tensor]]:
        return (
            [column[idx] for column in self.columns],
            self.labels[idx] if self.labels is not None else None,
        )


__pdoc__["TensorDataset.__len__"] = True
__pdoc__["TensorDataset.__getitem__"] = True


class MultipleOutputWrapper(Module):
    """Utility wrapper to select one output of a model with multiple outputs.

    Args:
        module: A model with more than one outputs.
        output: Index of the output to retain.
    """

    def __init__(self, module: Module, output: int = 0) -> None:
        super().__init__()
        self.inner = module
        self.output = output

    def forward(self, *args, **kwargs) -> Tensor:
        output = self.inner.forward(*args, **kwargs)
        return output[self.output]


__all__ = ["TensorDataset", "MultipleOutputWrapper"]
