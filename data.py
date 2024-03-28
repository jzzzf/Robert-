import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
# 读取训练集
train_df = pd.read_csv('you_data_path', sep='\t', header=None)
valid_df = pd.read_csv('you_data_path', sep='\t', header=None)
test_df = pd.read_csv('you_data_path', sep='\t', header=None)
train_df.columns = ['text', 'label']
valid_df.columns = ['text', 'label']
test_df.columns = ['text']


class MyDataset(Dataset):
    def __init__(self, data, tokenizer, mode='train', device='cuda'):
        """
        初始化数据集。
        :param data: 包含评论和标签的DataFrame或其他类似数据结构。
        :param tokenizer: 用于文本编码的分词器。
        :param mode: 数据集模式，'train'、'test' 或其他。
        :param device: 数据应被发送到的设备，通常是'cpu'或'cuda'（如果可用）。
        """
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode
        self.device = device
        # 检查CUDA是否可用，如果指定了'cuda'但不可用，则回退到'cpu'
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'

    def __len__(self):
        """返回数据集的大小。"""
        return len(self.data)

    def __getitem__(self, index):
        """
        返回给定索引处的样本，并确保它在正确的设备上。
        :param index: 样本索引。
        """
        # 获取并编码文本
        text = self.data.loc[index, 'text']
        encoded_text = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=32,
            return_tensors='pt'
        )

        input_ids = encoded_text['input_ids'].to(self.device).squeeze()
        attention_mask = encoded_text['attention_mask'].to(self.device).squeeze()

        # 根据模式返回样本
        if self.mode == 'train':
            label = self.data.loc[index, 'label']
            label_tensor = torch.tensor(label, device=self.device)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label_tensor
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }



