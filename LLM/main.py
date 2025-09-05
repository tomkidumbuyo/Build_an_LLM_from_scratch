import os
import tiktoken
import urllib

from Dataset import GPTDataset
from torch.utils.data import DataLoader


def create_dataloader_v1(
    txt, 
    batch_size=4, 
    max_length=256,
    stride=128, 
    shuffle=True, 
    drop_last=True,
    num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

file_path = "assets/the-verdict.txt"
url = (
 "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
 "main/ch02/01_main-chapter-code/the-verdict.txt"
)

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        raw_text = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(raw_text)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
features, labels = next(data_iter)
print('features', features)
print('labels', labels)

