import torch

# tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
# tokenizer.save_pretrained('./tests/models/')
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', './tests/models')
# model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
# model.save_pretrained('./tests/models/')
model = torch.hub.load('huggingface/pytorch-transformers', 'model', './tests/models')

text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"

indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)

segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

# with torch.no_grad():
#     encoded_layers, _ = model(tokens_tensor, token_type_ids=segments_tensors)

# print(encoded_layers)
# print(model)
s = torch.jit.trace(model, example_inputs={ 'forward': tokens_tensor })
