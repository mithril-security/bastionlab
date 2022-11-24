from torchvision.models import efficientnet_b0
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor, Normalize, Compose
from bastionai.client import Connection
from time import time

transform = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    lambda x: [x.squeeze(0)],
])

train_dataset = CIFAR100("data", train=True, transform=transform, download=True)
test_dataset = CIFAR100("data", train=False, transform=transform, download=True)

model = efficientnet_b0()

with Connection("localhost", 50051) as client:
    remote_dataset = client.RemoteDataset(train_dataset, test_dataset, name="CIFAR100")

    remote_learner = client.RemoteLearner(
        model,
        remote_dataset,
        max_batch_size=64,
        loss="cross_entropy",
        model_name="EfficientNet-B0",
        # expand=False,
        device="cuda:0",
    )

    start = time()
    remote_learner.fit(nb_epochs=1, eps=6.0)
    print(time() - start)
    remote_learner.test(metric="accuracy")
