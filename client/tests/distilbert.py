from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from bastionai.client import Connection
from bastionai.psg import expand_weights
from bastionai.utils import MultipleOutputWrapper, TensorDataset
from bastionai.pb.remote_torch_pb2 import TestConfig, TrainConfig, Empty
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# file_path = "./tests/data/SMSSpamCollection"

# Load data
# df = pd.DataFrame({ "label": int(), "text": str() }, index = [])
# with open(file_path) as f:
#   for line in f.readlines():
#     split = line.split('\t')
#     df = df.append({
#         "label": 1 if split[0] == "spam" else 0,
#         "text": split[1]
#         }, ignore_index = True)

# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Preprocessing
# token_id = []
# attention_masks = []
# for sample in df.text.values:
#     encoding_dict = tokenizer.encode_plus(
#         sample,
#         add_special_tokens = True,
#         max_length = 32,
#         pad_to_max_length = True,
#         return_attention_mask = True,
#         return_tensors = 'pt'
#     )
#     token_id.append(encoding_dict['input_ids']) 
#     attention_masks.append(encoding_dict['attention_mask'])

# token_id = torch.cat(token_id, dim = 0)
# attention_masks = torch.cat(attention_masks, dim = 0)
# labels = torch.tensor(df.label.values)

# torch.save(token_id, "tests/data/token_id.pt")
# torch.save(attention_masks, "tests/data/attention_masks.pt")
# torch.save(labels, "tests/data/labels.pt")

token_id = torch.load("tests/data/token_id.pt")
attention_masks = torch.load("tests/data/attention_masks.pt")
labels = torch.load("tests/data/labels.pt")

# Make training and testing datasets
val_ratio = 0.2
batch_size = 4

train_idx, test_idx = train_test_split(
    np.arange(len(labels)),
    test_size=val_ratio,
    shuffle=True,
    stratify=labels
)
train_idx = train_idx[:160]

train_set = TensorDataset([
    token_id[train_idx], 
    attention_masks[train_idx]
], labels[train_idx])

test_set = TensorDataset([
    token_id[test_idx], 
    attention_masks[test_idx]
], labels[test_idx])


# Load, expand and trace model
model = DistilBertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    torchscript=True
)
expand_weights(model, batch_size)
# print(model)
# raise Exception("Stop")

[text, mask], label = train_set[0]
traced_model = torch.jit.trace(
    MultipleOutputWrapper(model, 0),
    [
        text.unsqueeze(0),
        mask.unsqueeze(0),
        #label.unsqueeze(0)
    ]
)

# from torch.utils.data import DataLoader
# d = DataLoader(train_set, batch_size=16)
# for [text, mask], label in d:
#     out = traced_model(text, mask)
#     loss = torch.nn.functional.cross_entropy(out, label)
#     loss.backward()
    # raise Exception("Stop")

with Connection("::1", 50051) as client:
    model_ref = client.send_model(
        traced_model,
        "Expanded DistilBERT",
        b"secret"
    )
    print(f"Model ref: {model_ref}")

    train_dataset_ref = client.send_dataset(
        train_set,
        "SMSSpamCollection",
        b'secret'
    )
    print(f"Dataset ref: {train_dataset_ref}")

    client.train(TrainConfig(
        model=model_ref,
        dataset=train_dataset_ref,
        batch_size=batch_size,
        epochs=2,
        device="cpu",
        metric="cross_entropy",
        differential_privacy=TrainConfig.DpParameters(
            max_grad_norm=100.,
            noise_multiplier=0.001
        ),
        # standard=Empty(),
        adam=TrainConfig.Adam(
            learning_rate=5e-5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            weight_decay=0,
            amsgrad=False
        )
    ))
