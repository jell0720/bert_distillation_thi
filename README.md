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
git clone http://gitlab.thi.com.tw/jell/bert_distillation.git
cd bert-distillation
```

### 2. 教師模型訓練

執行指令如下：

```bash
python trainer_teacher.py --data_path ./data/sst2 --output_dir ./models/teacher
```

### 3. 學生模型蒸餾

執行以下指令：

```bash
python distilled-bert.py --teacher_model ./models/teacher --output_dir ./models/student
```

---

## 蒸餾演算法詳細說明

### 1. **軟標籤 (Soft Label) 與溫度 (Temperature)**

- 從教師模型 (bert-large-uncased) 輸出 logits，經由「溫度參數」$T$ 進行縮放，計算出軟標籤。
- 公式：
  $$
  p_{teacher}(y \mid x) = \mathrm{softmax}\left(\frac{z_{teacher}}{T}\right)
  $$
  其中 $z_{teacher}$ 為教師模型 logits。
- 當 $T>1$，分布更平滑，學生模型可學習更細微的資訊。

### 2. **KL Divergence 損失**

- 學生模型 (bert-base-uncased) 產生 logits，同樣使用溫度 $T$ 產生軟標籤。
- 與教師的軟標籤透過 KL 散度 (Kullback-Leibler Divergence) 進行對齊。
- 目標為最小化：
  $$
  L_{KL} = \sum_y p_{teacher}(y \mid x) \log \frac{p_{teacher}(y \mid x)}{p_{student}(y \mid x)}
  $$

### 3. **硬標籤 (Hard Label) 交叉熵損失**

- 學生模型同時學習原始資料的真實標籤 (0/1)。
- 損失函數結合軟標籤和硬標籤：
  $$
  L_{total} = \alpha \cdot L_{KL} + (1 - \alpha) \cdot L_{CE}
  $$
  其中 $L_{CE}$ 是交叉熵損失。

---

## 評估與應用

執行模型評估：
```bash
python evaluate.py --model ./models/student --data_path ./data/sst2/test
```

---

## 常見問題與解決方案

### 1. 資料集下載

```python
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/sst2")
dataset.save_to_disk("./data/sst2")
```

### 2. 模型下載失敗

```bash
huggingface-cli download bert-large-uncased -d ./models/teacher
```

---

## 授權條款

本專案採用 **MIT License**，詳細內容請參閱 [LICENSE](./LICENSE)。

---

## 聯絡方式

如有任何問題，請聯絡 [jell@thi.com.tw](mailto:jell@thi.com.tw)。

