import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import DataLoader

from bastionlab import Connection
from bastionlab.torch.optimizer_config import Adam
from bastionlab.torch.utils import MultipleOutputWrapper, TensorDataset

##########################################################################################

file_path = "./SMSSpamCollection"

labels = []
texts = []
with open(file_path) as f:
    for line in f.readlines():
        split = line.split("\t")
        labels.append(1 if split[0] == "spam" else 0)
        texts.append(split[1])
df = pd.DataFrame({"label": labels, "text": texts})

################################################################################

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

token_id = []
attention_masks = []
for sample in df.text.values:
    encoding_dict = tokenizer.encode_plus(
        sample,
        add_special_tokens=True,
        max_length=32,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    token_id.append(encoding_dict["input_ids"])
    attention_masks.append(encoding_dict["attention_mask"])

token_id = torch.cat(token_id, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df.label.values)

######################################################################

val_ratio = 0.2

train_idx, test_idx = train_test_split(
    np.arange(len(labels)), test_size=val_ratio, shuffle=True, stratify=labels
)

print(token_id[train_idx])


train_set = TensorDataset(
    [token_id[train_idx], attention_masks[train_idx]], labels[train_idx]
)

test_set = TensorDataset(
    [token_id[test_idx], attention_masks[test_idx]], labels[test_idx]
)

#######################################################################

# Do not display warnings about layer not initialized
# with pretrained weights (classification layers, this is fine)
from transformers import logging

logging.set_verbosity_error()

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    torchscript=True,
)
model = MultipleOutputWrapper(model, 0)

###############################################################################################

# The Data Owner privately uploads their model online
client = Connection("localhost").client.torch
remote_dataset = client.RemoteDataset(train_set, test_set, name="SMSSpamCollection")

################################################################################################

client = Connection("localhost").client.torch
remote_datasets = client.list_remote_datasets()

print([str(ds) for ds in remote_datasets])
print(remote_datasets[0].trace_input)

#################################################################################################

import warnings

warnings.filterwarnings("ignore")

# The Data Scientist discovers available datasets and use one of them to train their model
client = Connection("localhost").client.torch
remote_learner = client.RemoteLearner(
    model,
    remote_datasets[0],
    max_batch_size=2,
    loss="cross_entropy",
    optimizer=Adam(lr=5e-5),
    model_name="DistilBERT",
)

remote_learner.fit(nb_epochs=1, eps=6.0)  # , poll_delay=1.0)
remote_learner.test(metric="accuracy")

trained_model = remote_learner.get_model()
