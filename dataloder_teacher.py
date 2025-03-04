# 安裝必要套件（若尚未安裝）
# pip install transformers datasets torch accelerate scikit-learn

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

# -----------------------------------------------------
# 1. 設定裝置（優先使用 GPU，其次 MPS，最後 CPU）
# -----------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"🔥 使用的裝置: {device}")

# -----------------------------------------------------
# 2. 載入 SST-2 數據集（增加快取功能）
# -----------------------------------------------------
cache_dir = "./dataset_cache"
os.makedirs(cache_dir, exist_ok=True)
dataset = load_dataset("stanfordnlp/sst2", cache_dir=cache_dir)
train_data = dataset["train"]
val_data = dataset["validation"]

# -----------------------------------------------------
# 3. 預處理數據（Tokenization 優化）
# -----------------------------------------------------
model_name = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    # 動態填充而非固定長度，提高效率
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding=False,  # 在 DataCollator 中進行動態填充
        max_length=128
    )

# 使用多進程加速數據預處理
train_data = train_data.map(
    preprocess_function, 
    batched=True, 
    num_proc=4,  # 使用多進程加速
    desc="正在處理訓練數據"
)
val_data = val_data.map(
    preprocess_function, 
    batched=True, 
    num_proc=4,
    desc="正在處理驗證數據"
)

# 設定資料格式為 PyTorch tensors，並移除不必要的列以節省記憶體
columns = ["input_ids", "attention_mask", "label"]
train_data.set_format(type="torch", columns=columns)
val_data.set_format(type="torch", columns=columns)

# -----------------------------------------------------
# 4. 載入 BERT-Large 模型（增加記憶體優化選項）
# -----------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32  # 只在 CUDA 設備上使用半精度浮點數
)
model.to(device)

# -----------------------------------------------------
# 5. 定義更全面的評估指標函式
# -----------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='weighted'),
        "precision": precision_score(labels, predictions, average='weighted'),
        "recall": recall_score(labels, predictions, average='weighted')
    }

# -----------------------------------------------------
# 6. 設定訓練參數與建構 Trainer 物件（優化訓練設定）
# -----------------------------------------------------
training_args = TrainingArguments(
    output_dir="./bert-large-sst2",
    eval_strategy="epoch",   # 每個 epoch 結束後做驗證
    save_strategy="epoch",         # 每個 epoch 結束後儲存模型
    per_device_train_batch_size=8,  # 增加批次大小，提高訓練效率
    per_device_eval_batch_size=16,  # 評估時可使用更大的批次
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    optim="adamw_torch",
    logging_dir='./logs',
    logging_steps=50,
    fp16=device.type == "cuda",  # 只在 CUDA 設備上啟用混合精度訓練
    gradient_accumulation_steps=2,  # 梯度累積，可使用更大的有效批次大小
    warmup_ratio=0.1,  # 學習率預熱
    load_best_model_at_end=True,  # 訓練結束後載入最佳模型
    metric_for_best_model="accuracy",  # 以準確率為標準選擇最佳模型
    save_total_limit=2,  # 只保存最近的兩個檢查點，節省磁碟空間
)

# 使用動態填充的數據整理器，提高記憶體使用效率
data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -----------------------------------------------------
# 7. 訓練與評估模型
# -----------------------------------------------------
trainer.train()
eval_result = trainer.evaluate()
print(f"評估結果: {eval_result}")

# -----------------------------------------------------
# 8. 儲存已訓練模型與 tokenizer
# -----------------------------------------------------
model.save_pretrained("fine-tuned-bert-large")
tokenizer.save_pretrained("fine-tuned-bert-large")
print("✅ 模型已儲存至 'fine-tuned-bert-large'")

# -----------------------------------------------------
# 9. 釋放記憶體
# -----------------------------------------------------
del model
del trainer
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("🧹 已釋放記憶體資源")