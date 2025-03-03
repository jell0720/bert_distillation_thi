# BERT 蒸餾示範專案

## 專案簡介

本專案示範如何使用 **bert-large-uncased** 模型，在 **SST-2** (Stanford Sentiment Treebank) 資料集上進行 **fine-tune**，並透過知識蒸餾技術訓練 **bert-base-uncased** 學生模型，達到更輕量化的應用。

### 主要特點
- **知識蒸餾**：利用大模型（教師）訓練較小的模型（學生）。
- **效能優化**：減少計算資源需求，適合實際部署。
- **簡單易用**：透過 Python 腳本即可快速執行完整流程。

---

## 專案結構

```
📂 bert-distillation
├── distilled-bert.py        # 模型蒸餾程式
├── trainer_teacher.py       # 教師模型訓練程式
├── pyproject.toml           # 依賴管理檔案
├── requirements.txt         # 依賴安裝檔案（選擇性）
├── LICENSE                  # 授權條款
├── README.md                # 本說明文件
└── data/                    # 訓練與測試資料集（請手動下載）
```

---

## 環境需求

- Python 3.7 以上
- `torch`, `transformers`, `datasets`, `accelerate`
- **建議使用 Poetry 來管理套件依賴**

安裝 Poetry：

```bash
pip install poetry
```

安裝專案依賴：

```bash
poetry install
```

或者使用 pip：

```bash
pip install -r requirements.txt
```

---

## 訓練流程

### 1. 複製專案

```bash
git clone <專案網址>
cd bert-distillation
```

### 2. 訓練教師模型

- 使用 **bert-large-uncased** 於 **SST-2** 進行 **fine-tune**。
- 完成後產生教師模型的權重。

執行以下指令：

```bash
python trainer_teacher.py --data_path ./data/sst2 --output_dir ./models/teacher
```

### 3. 執行知識蒸餾

- 讓 **bert-base-uncased** 學生模型學習教師模型知識。
- 調整蒸餾超參數以獲得最佳效果。

執行以下指令：

```bash
python distilled-bert.py --teacher_model ./models/teacher --output_dir ./models/student
```

---

## 評估與應用

- **模型評估**：
  
  ```bash
  python evaluate.py --model ./models/student --data_path ./data/sst2/test
  ```
- **應用部署**：
  - 優化後的學生模型適合嵌入輕量應用。
  - 可搭配 ONNX Runtime 或 TensorRT 進一步加速推理。

---

## 常見問題與解決方案

### 1. 資料集下載
請確保 **SST-2** 資料集已下載並存放於 `./data/sst2` 目錄內。
可透過 `datasets` 套件下載：

```python
from datasets import load_dataset
dataset = load_dataset("glue", "sst2")
dataset.save_to_disk("./data/sst2")
```

### 2. 模型下載失敗
如 Hugging Face 下載模型失敗，可先手動下載並存放於 `./models` 內，然後指定 `--teacher_model` 參數。

```bash
huggingface-cli download bert-large-uncased -d ./models/teacher
```

---

## 貢獻指南

- **Issue 回報**：發現錯誤或改進建議，請提交 GitHub Issue。
- **Pull Request**：修正 Bug 或新增功能，請發 PR，並確保符合 PEP8 編碼風格。

---

## 授權條款

本專案採用 **MIT License**，詳細內容請參閱 [LICENSE](./LICENSE)。

---

## 聯絡方式

如有任何問題，請聯絡 [jell@thi.com.tw](mailto:jell@thi.com.tw)。
