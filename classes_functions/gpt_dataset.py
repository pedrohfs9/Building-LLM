from torch.utils.data import DataLoader, Dataset
import torch

class GPTdatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        tokens_ids = tokenizer.encode(txt, allowed_special={"|<endoftext|>"})
        assert len(tokens_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        for i in range(0, len(tokens_ids) - max_length, stride):
            input_chunk = tokens_ids[i:i + max_length]
            target_chunk = tokens_ids[i + 1: i+ max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
