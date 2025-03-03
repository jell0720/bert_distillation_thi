# 安裝必要套件（若尚未安裝）
# pip install transformers datasets torch accelerate

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# -----------------------------------------------------
# 1. 設定 MPS 作為裝置（macOS M2/M3）
# -----------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔥 使用的裝置: {device}")

# -----------------------------------------------------
# 2. 載入 SST-2 數據集（GLUE benchmark）
# -----------------------------------------------------
dataset = load_dataset("glue", "sst2")
train_data = dataset["train"]
val_data = dataset["validation"]

# -----------------------------------------------------
# 3. 預處理數據（Tokenization）
# -----------------------------------------------------
model_name = "bert-large-uncased"

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 定義 Tokenization 函數
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

# 應用 Tokenization
train_data = train_data.map(preprocess_function, batched=True)
val_data = val_data.map(preprocess_function, batched=True)

# 設定格式以適應 Hugging Face `Trainer`
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------------------------------------
# 4. 載入 BERT-Large 模型（並移動到 MPS）
# -----------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# -----------------------------------------------------
# 5. 設定訓練參數（更新 `eval_strategy`）
# -----------------------------------------------------
training_args = TrainingArguments(
    output_dir="./bert-large-sst2",   # 模型儲存位置
    eval_strategy="epoch",            # ✅ 修正 `evaluation_strategy` 已棄用
    save_strategy="epoch",
    per_device_train_batch_size=4,    # M2 記憶體小，建議降低 batch size
    per_device_eval_batch_size=4,
    num_train_epochs=3,               # 訓練 3 個 Epoch
    learning_rate=2e-5,               # 學習率
    weight_decay=0.01,                # L2 正則化
    optim="adamw_torch",              # ✅ 使用 PyTorch 內建 AdamW 優化器
    logging_dir="./logs",             # 記錄日誌
    logging_steps=50,                 # 每 50 步記錄一次
    report_to="none"                   # 避免自動上傳到 Hugging Face
)

# -----------------------------------------------------
# 6. 設定 `Trainer` 進行微調（Fine-tuning）
# -----------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# 開始訓練
trainer.train()

# -----------------------------------------------------
# 7. 測試模型（驗證準確率）
# -----------------------------------------------------
results = trainer.evaluate()
print(f"Fine-tuned BERT-Large 準確率: {results['eval_accuracy']:.4f}")

# -----------------------------------------------------
# 8. 儲存已訓練的模型
# -----------------------------------------------------
model.save_pretrained("fine-tuned-bert-large")
tokenizer.save_pretrained("fine-tuned-bert-large")

print("✅ 模型已儲存至 'fine-tuned-bert-large'")