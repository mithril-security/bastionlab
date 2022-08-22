import pytest
from bastionai.utils import *
from bastionai.pb.remote_torch_pb2 import ClientInfo
from torch.nn.functional import relu
from torch.nn import Module, Linear
from transformers import DistilBertForSequenceClassification


class DummyModule(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(1, 1)
        self.fc2 = Linear(1, 1)
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x


class Params(Module):
    def __init__(self, model: Module) -> None:
        super().__init__()
        for name, value in model.named_parameters():
            setattr(self, name.replace(".", "_"), Parameter(value))

    def forward(self, x: Tensor) -> Tensor:
        return x


EPS = 1e-8
empty_client_info = ClientInfo()


def tensor_eq(a: Tensor, b: Tensor) -> bool:
    return ((a - b)**2).sum() < EPS


def module_eq(a: Module, b: Module, test_input: List[Tensor]):
    params1 = list(a.named_parameters())
    params1.sort(key=lambda x: x[0])
    params2 = list(b.named_parameters())
    params2.sort(key=lambda x: x[0])

    y1 = a(*test_input)
    y2 = b(*test_input)

    return (
        len(params1) == len(params2) and
        all([a[0] == b[0] and ((a[1] - b[1])**2).sum() < EPS for a, b in zip(params1, params2)]) and
        tensor_eq(y1, y2)
    )


@pytest.fixture
def simple_dataset() -> TensorDataset:
    X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
    Y = torch.tensor([[0.0], [2.0], [1.0], [0.4]])

    return TensorDataset([X], Y)


def test_data_wrapping(simple_dataset):
    [X1], Y1 = simple_dataset.columns, simple_dataset.labels
    wrapper = DataWrapper([X1], Y1)
    [X2], Y2 = data_from_wrapper(wrapper)

    assert tensor_eq(X1, X2) and tensor_eq(Y1, Y2)


@pytest.mark.parametrize("chunk_size, batch_size", [
    (100_000_000, 1024),
    (100_000_000, 2),
    (8, 1024),
    #(8, 2),
])
def test_simple_dataset_serialization(chunk_size, batch_size, simple_dataset):
    ds1 = simple_dataset
    chunks = serialize_dataset(
        ds1, description="", secret=b"", chunk_size=chunk_size, batch_size=batch_size, client_info=empty_client_info)
    ds2 = dataset_from_chunks(chunks)

    assert len(ds1) == len(ds2)
    assert all([x == y for x, y in zip(ds1, ds2)])


@pytest.fixture
def sms_spam_collection() -> TensorDataset:
    token_id = torch.load("tests/token_id.pt")
    attention_masks = torch.load("tests/attention_masks.pt")
    labels = torch.load("tests/labels.pt")

    return TensorDataset([
        token_id,
        attention_masks
    ], labels)


def test_real_dataset_serialization(sms_spam_collection):
    ds1 = sms_spam_collection
    chunks = serialize_dataset(
        ds1, description="", secret=b"", batch_size=10_000, client_info=empty_client_info)
    ds2 = dataset_from_chunks(chunks)

    assert all([
        tensor_eq(x1[0], x2[0]) and
        tensor_eq(x1[1], x2[1]) and
        y1 == y2
        for (x1, y1), (x2, y2) in zip(ds1, ds2)
    ])


def run_model_test(model: Module, chunk_size: int, test_input: List[Tensor]):
    model1 = model
    chunks = serialize_model(
        model1, description="", secret=b"", chunk_size=chunk_size, client_info=empty_client_info)
    models = list(unstream_artifacts(
        (chunk.data for chunk in chunks), deserialization_fn=torch.jit.load
    ))
    assert len(models) == 1
    model2 = models[0]

    assert module_eq(model1, model2, test_input)


@pytest.mark.parametrize("chunk_size", [100_000_000, 8])
def test_simple_model_serialization(chunk_size):
    run_model_test(DummyModule(), chunk_size, [torch.randn(1)])


def test_real_model_serialization(sms_spam_collection):
    from transformers import logging
    logging.set_verbosity_error()
    import warnings
    warnings.filterwarnings("ignore")

    model = DistilBertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
        torchscript=True
    )
    model = MultipleOutputWrapper(model, 0)

    inputs = [x.unsqueeze(0) for x in sms_spam_collection[0][0]]
    traced_model = torch.jit.trace(
        model,
        inputs
    )
    run_model_test(traced_model, 100_000_000, inputs)


def test_simple_model_deserialization():
    model1 = DummyModule()
    model2 = DummyModule()

    chunks = serialize_model(Params(model1), description="", secret=b"",
                             client_info=empty_client_info)
    deserialize_weights_to_model(model2, chunks)

    assert module_eq(model1, model2, [torch.randn(1)])
