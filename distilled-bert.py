# 安裝必要套件（若尚未安裝，可使用 pip 安裝）
# pip install transformers datasets torch accelerate tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm  # 進度條

# -----------------------------------------------------
# 1. 設定 MPS 或 CUDA 作為裝置
# -----------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用的裝置: {device}")

# -----------------------------------------------------
# 2. 載入已微調的教師模型與學生模型
# -----------------------------------------------------
# Teacher Model: 已微調的 BERT-Large（SST-2），如果是trainer_teacher.py 應該是fine-tuned-bert-large
teacher_model_name = "yoshitomo-matsubara/bert-large-uncased-sst2"
student_model_name = "bert-base-uncased"

# 載入教師模型
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name, num_labels=2).to(device)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# 載入學生模型
student_model = AutoModelForSequenceClassification.from_pretrained(student_model_name, num_labels=2).to(device)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# 設定教師模型為推論模式（不進行訓練）
teacher_model.eval()

# -----------------------------------------------------
# 3. 載入並預處理 SST-2 數據集
# -----------------------------------------------------
dataset = load_dataset("stanfordnlp/sst2")
train_data = dataset["train"]
val_data = dataset["validation"]

# Tokenization 處理
def preprocess_function(examples):
    return teacher_tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

# 應用 Tokenization
train_data = train_data.map(preprocess_function, batched=True)
val_data = val_data.map(preprocess_function, batched=True)

# 設定格式以適應 PyTorch
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------------------------------------
# 4. 計算教師模型的 Soft Labels
# -----------------------------------------------------
# 建立 DataLoader，批次大小 16
train_loader = DataLoader(train_data, batch_size=16, shuffle=False)

def get_teacher_logits(model, dataloader):
    logits_list = []
    with torch.no_grad():
        # 加入 tqdm 進度條
        for batch in tqdm(dataloader, desc="計算教師 Soft Labels"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_list.append(outputs.logits)
    return torch.cat(logits_list)

# 取得教師模型預測的 logits 與 Soft Labels
teacher_logits = get_teacher_logits(teacher_model, train_loader)
temperature = 2.0
teacher_soft_labels = F.softmax(teacher_logits / temperature, dim=-1)

# -----------------------------------------------------
# 5. 建立訓練用的 TensorDataset 與 DataLoader
# -----------------------------------------------------
# 重新組合訓練資料，加上 Soft Labels
train_tensor_dataset = TensorDataset(
    train_data["input_ids"],
    train_data["attention_mask"],
    train_data["label"],
    teacher_soft_labels
)

val_tensor_dataset = TensorDataset(
    val_data["input_ids"],
    val_data["attention_mask"],
    val_data["label"]
)

train_dataloader = DataLoader(train_tensor_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_tensor_dataset, batch_size=16, shuffle=False)

# -----------------------------------------------------
# 6. 定義蒸餾損失函數
# -----------------------------------------------------
def distillation_loss(student_logits, teacher_soft_labels, hard_labels, alpha=0.5, temperature=2.0):
    # KL 散度損失
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_soft_labels, reduction="batchmean")
    # 交叉熵損失
    ce_loss = F.cross_entropy(student_logits, hard_labels)
    return alpha * kl_loss + (1 - alpha) * ce_loss

# -----------------------------------------------------
# 7. 訓練學生模型
# -----------------------------------------------------
student_model.train()
optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)
num_epochs = 3
alpha = 0.5

for epoch in range(num_epochs):
    total_loss = 0
    # tqdm 顯示每個 epoch 的進度
    for batch in tqdm(train_dataloader, desc=f"訓練 Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids, attention_mask, labels, soft_labels = [b.to(device) for b in batch]
        outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = outputs.logits
        loss = distillation_loss(student_logits, soft_labels, labels, alpha=alpha, temperature=temperature)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, 平均損失: {avg_loss:.4f}")

# -----------------------------------------------------
# 8. 測試學生模型（驗證準確率）
# -----------------------------------------------------
student_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in tqdm(val_dataloader, desc="驗證學生模型"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print(f"學生模型在驗證集上的準確率: {accuracy:.4f}")

# -----------------------------------------------------
# 9. 儲存蒸餾後的學生模型
# -----------------------------------------------------
student_model.save_pretrained("distilled-bert")
student_tokenizer.save_pretrained("distilled-bert")
print("✅ 蒸餾後的 BERT 模型已儲存至 'distilled-bert'")