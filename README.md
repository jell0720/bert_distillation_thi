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

此階段使用 **bert-large-uncased** 在 **SST-2** 上做標準微調 (fine-tune)，過程如下：

1. **讀取預訓練模型**：
   - 載入 `bert-large-uncased`，並加上分類層 (linear layer)。
2. **設定超參數**：
   - 如 epoch、batch size、learning rate 等。
3. **前向傳遞與損失計算**：
   - 對每個 batch 輸入，模型輸出 logits。
   - 使用交叉熵損失 (Cross Entropy) 與真實標籤 (0/1) 計算 loss。
4. **反向傳遞與更新**：
   - 根據 loss 做反向傳遞，更新模型參數。
5. **評估與儲存**：
   - 透過驗證集 (validation set) 評估準確度，並儲存教師模型權重。

執行指令如下：

```bash
python trainer_teacher.py --data_path ./data/sst2 --output_dir ./models/teacher
```

### 3. 學生模型蒸餾

- 讓 **bert-base-uncased** 學生模型學習教師模型知識。
- 調整蒸餾超參數以獲得最佳效果。

執行以下指令：

```bash
python distilled-bert.py --teacher_model ./models/teacher --output_dir ./models/student
```

---

## 蒸餾演算法詳細說明

在 `distilled-bert.py` 及相關程式碼中，運用了以下技術來進行知識蒸餾：

1. **軟標籤 (Soft Label) 與溫度 (Temperature)**

   - 從教師模型 (bert-large-uncased) 輸出 logits，經由「溫度參數」\$T\$ 進行縮放，計算出軟標籤。
   - 公式範例： \(p_{teacher}(y\mid x) = \mathrm{softmax}\left(\frac{z_{teacher}}{T}\right)\) 其中 \$z\_{teacher}\$ 為教師模型 logits。
   - 當 \$T>1\$，分布更平滑，學生模型可從教師的機率分布中學到更細微的資訊。

2. **KL Divergence 損失**

   - 學生模型 (bert-base-uncased) 輸出 logits，同樣使用相同溫度 \$T\$ 產生軟標籤。
   - 與教師的軟標籤透過 KL 散度 (Kullback-Leibler Divergence) 進行對齊。
   - 目標為最小化： \(L_{KL} = \sum_y p_{teacher}(y\mid x) \log\frac{p_{teacher}(y\mid x)}{p_{student}(y\mid x)}\)

3. **硬標籤 (Hard Label) 交叉熵損失**

   - 除了與教師軟標籤對齊，學生模型也同時學習原始資料的真實標籤 (0/1)。
   - 損失函數通常會以權重 \$\alpha\$ 與 \$(1-\alpha)\$ 結合軟標籤和硬標籤： \(L_{total} = \alpha * L_{KL} + (1 - \alpha) * L_{CE}\)
   - 其中 \$L\_{CE}\$ 是學生模型輸出與真實標籤之間的交叉熵。

4. **蒸餾目標**

   - 透過同時學習教師模型分布及原始真實標籤，學生模型能在參數量較小的前提下保持接近教師模型的表現。
   - 適合在資源有限或需要減少推理延遲的情境。

此過程在 `distilled-bert.py` 內可看到主流程：

- **讀取教師模型** 的權重與配置。
- **初始化學生模型** (bert-base-uncased)。
- **計算教師 logits**：在每個 batch 上，先將輸入丟進教師模型取得 logits。
- **計算學生 logits**：同樣丟進學生模型得到輸出。
- **蒸餾損失計算**：根據上述 KL 散度與交叉熵公式，計算 loss。
- **反向傳播更新**：依照 loss 值對學生模型進行參數更新。

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

請確保 **SST-2** 資料集已下載並存放於 `./data/sst2` 目錄內。 可透過 `datasets` 套件下載：

```python
from datasets import load_dataset
dataset = load_dataset(stanfordnlp/sst2")
dataset.save_to_disk("./data/sst2")
```

### 2. 模型下載失敗

如 Hugging Face 下載模型失敗，可先手動下載並存放於 `./models` 內，然後指定 `--teacher_model` 參數。

```bash
huggingface-cli download bert-large-uncased -d ./models/teacher
```

---

## 授權條款

本專案採用 **MIT License**，詳細內容請參閱 [LICENSE](./LICENSE)。

---

## 聯絡方式

如有任何問題，請聯絡 [jell@thi.com.tw](mailto:jell@thi.com.tw)。

