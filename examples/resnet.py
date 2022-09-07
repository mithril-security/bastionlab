from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor, Normalize, Compose
from bastionai.client import Connection

transform = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    lambda x: [x],
])

train_dataset = CIFAR100("data", train=True, transform=transform, download=True)
test_dataset = CIFAR100("data", train=False, transform=transform, download=True)

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

with Connection("localhost", 50051) as client:
    remote_dataloader = client.RemoteDataset(train_dataset, test_dataset, name="CIFAR100")

with Connection("localhost", 50051) as client:
    remote_learner = client.RemoteLearner(
        model,
        remote_dataloader,
        max_batch_size=4,
        loss="cross_entropy",
        model_name="ResNet50",
    )

    remote_learner.fit(nb_epochs=100, eps=6.0)
    # remote_learner.test(metric="accuracy")
    
    # trained_model = remote_learner.get_model()
