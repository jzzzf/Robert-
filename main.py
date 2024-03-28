from data import MyDataset
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader
from models import RoBERT, RoBertLSTM, ErnieLSTM
from lossfn import MultiFocalLoss
from transformers import get_cosine_schedule_with_warmup
from torch.optim import Adam
from sklearn.metrics import f1_score
from tqdm import tqdm
model_path = 'you_model_path'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)

tokenizer = AutoTokenizer.from_pretrained(model_path)
# 读取训练集
train_df = pd.read_csv('you_data_path', sep='\t', header=None)
valid_df = pd.read_csv('you_data_path', sep='\t', header=None)
test_df = pd.read_csv('you_data_path', sep='\t', header=None)
train_df.columns = ['text', 'label']
valid_df.columns = ['text', 'label']
test_df.columns = ['text']

train_data = train_df.reset_index(drop=True)
valid_data = valid_df.reset_index(drop=True)
valid_label = valid_data['label']
valid_label = valid_label.values.tolist()

test_data = test_df.copy()

# 调用MyDataset
# 训练集
train_dataset = MyDataset(train_data, tokenizer=tokenizer, mode='train', device=device)
# 验证集
valid_dataset = MyDataset(valid_data, tokenizer=tokenizer, mode='train', device=device)
# 测试集
test_dataset = MyDataset(test_data, tokenizer=tokenizer, mode='test', device=device)

# 调用 dataloader
# 批处理大小
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型
model = RoBERT().to(device)
model_name = 'Roberta'

# 定义损失函数，优化器
num_epochs = 15

# 交叉熵损失函数
# loss_fn=nn.CrossEntropyLoss()
# focal loss

loss_fn = MultiFocalLoss(num_class=6, alpha=[0.25, 0.06, 0.12, 0.07, 0.23, 0.27])
# 优化器
optimizer = Adam(model.parameters(), lr=5e-5)

total_steps = num_epochs * len(train_dataloader)

# 学习率调度器
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_training_steps=total_steps,
                                            num_warmup_steps=total_steps * 0.1)

# 训练模型
train_losses = []
valid_losses = []
Best_acc = 0
Best_F1 = 0
step = 0
loss_sum = 0

for epoch in tqdm(range(num_epochs)):
    model.train()  # 设置为训练模式
    for batch in tqdm(train_dataloader):
        step += 1

        # 正常训练
        out = model(batch['input_ids'], batch['attention_mask'])
        loss = loss_fn(out, batch['label'])
        loss_sum += loss.item()
        loss.backward()  # 正向传播

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"epoch：{epoch + 1}，平均训练损失：{loss_sum / 50}")
            loss_sum = 0

    # 验证集上进行评估
    model.eval()  # 设置为评估模式
    valid_loss_sum = 0
    correct = 0
    total = 0
    preds = []
    labels = []

    with torch.no_grad():  # 禁用梯度计算
        for val_batch in tqdm(valid_dataloader):
            outputs = model(val_batch['input_ids'], val_batch['attention_mask'])
            predicted_labels = torch.argmax(outputs, 1)
            correct += (predicted_labels == val_batch['label']).sum()
            total += val_batch['label'].size(0)
            preds.extend(list(predicted_labels.cpu().numpy()))
            labels.extend(list(val_batch['label'].cpu().numpy()))

    f1 = f1_score(valid_label, preds, average="macro")

    accuracy = correct / total
    valid_losses.append(accuracy)
    # 检查是否是当前最佳准确率
    if Best_acc < accuracy:
        Best_acc = accuracy
        torch.save(model.state_dict(), f'{model_name}/model_{Best_acc}.bin')
    # 检查是否是当前最佳f1得分
    if Best_F1 < f1:
        Best_F1 = f1
    print(f"epoch：{epoch + 1}，验证集准确率：{accuracy}，最高准确率：{Best_acc},验证集f1得分:{f1},最高f1得分:{Best_F1}")
