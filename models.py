import torch.nn as nn
from transformers import AutoModel, AutoConfig


class RoBERT(nn.Module):
    def __init__(self, num_classes,model_path):
        super(RoBERT, self).__init__()
        # 加载 RoBERTa 预训练模型的配置
        config = AutoConfig.from_pretrained(model_path)
        # 加载 RoBERTa 预训练模型
        self.bert = AutoModel.from_pretrained(model_path, config=config)
        # 定义分类层
        self.out = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # 获取 RoBERTa 的输出
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)

        pooler = bert_out['pooler_output']

        # 通过分类层进行分类
        out = self.out(pooler)

        return out


class RoBertLSTM(nn.Module):
    def __init__(self, num_classes,model_path):
        super(RoBertLSTM, self).__init__()
        # 加载 RoBERTa 预训练模型的配置
        config = AutoConfig.from_pretrained(model_path)
        # 加载 RoBERTa 预训练模型
        self.bert = AutoModel.from_pretrained(model_path, config=config)

        # 加入LSTM
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.out = nn.Linear(config.hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        # 获取 RoBERTa 的输出
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)

        last_hidden_state = bert_out.last_hidden_state
        result_out, _ = self.lstm(last_hidden_state)

        out = self.out(result_out[:, -1, :])
        return out


class ErnieLSTM(nn.Module):
    def __init__(self, num_classes,model_path):
        super(ErnieLSTM, self).__init__()
        # 加载 ERNIE预训练模型的配置
        config = AutoConfig.from_pretrained(model_path)
        # 加载 ERNIE预训练模型
        self.bert = AutoModel.from_pretrained(model_path, config=config)

        # 加入LSTM
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.out = nn.Linear(config.hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        # 获取 RoBERTa 的输出
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)

        last_hidden_state = bert_out.last_hidden_state
        result_out, _ = self.lstm(last_hidden_state)

        out = self.out(result_out[:, -1, :])
        return out



