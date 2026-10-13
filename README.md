# PolyPulse

**AI-Powered Prediction Market Intelligence** — HKUST ISOM5240 Deep Learning business application.

PolyPulse helps users evaluate the likely *demand category* (Low / Medium / High) for a proposed Polymarket question and investigate observed trading volumes from 20,000 similar historical markets. It **does not** predict the event outcome, guarantee future trading volume, or make trading/launch recommendations.

## Features

- **Demand:** Fine-tuned DeBERTa-v3-base Hugging Face `text-classification` pipeline, CPU-only. The exact text contract is `Question: {question.strip()} [SEP] Event: {event_title.strip()}`. Label order comes from the model's standard Hugging Face `config.json` (`Low`, `Medium`, `High`). Sequence length uses an explicit published input limit if present, otherwise the model/tokenizer maximum. If validation adjustments or USD-category cutoffs are not published, raw classification scores remain available and missing metadata is clearly indicated rather than invented.
- **Comparables:** Pretrained all-MiniLM-L6-v2 Hugging Face `feature-extraction` pipeline, CPU-only. Reuses the supplied `historical_search.py`, the existing 384-dimensional vectors, and cosine similarity to find up to 10 historical markets and their *actual stored* trading volumes in USD.
- **Historical range:** Median and 25th/75th percentiles of the returned sufficiently similar markets only. These statistics disappear if fewer than five comparables meet the threshold. With zero matches, the comparable-markets section simply says **No historical comparable markets found** (no empty table or chart).
- **Three tabs:** Market Analysis, 🔥 Popular Questions, and Model Performance. The Popular Questions tab uses real examples already present in the historical data, with emoji categories, category filters, nearest-neighbor counts, and median historical volumes. Editing the question/event title **preserves** the previous analysis; **Analyse Market** replaces it and **Clear** resets inputs and results.
- **Independent branches:** Either pipeline may fail without suppressing the other's results. Measured inference durations (excluding initial model load) are displayed.

## Project layout

```text
PolyPulse/
├── app.py
├── demand_prediction.py
├── historical_search.py
├── data/
│   ├── historical_markets.parquet
│   ├── market_embeddings.npy
│   └── metadata.json
├── results/
│   └── model_comparison.csv
├── tests/
│   ├── test_app.py
│   └── test_live_models.py
├── requirements.txt
├── README.md
└── .gitignore
```

`historical_search.py` is copied from the supplied original module. `data/` contains the **corrected** uploaded archive (20,000 records), not the earlier one-record archive. The parquet records are approximately 2 MB and precomputed embeddings approximately 31 MB uncompressed, small enough for normal GitHub hosting.

## Setup and run locally

Use Python 3.11 for consistent CPU dependencies. From this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

First model use requires internet: Hugging Face downloads the fine-tuned DeBERTa model and the pretrained MiniLM weights **to its model cache**, not into this GitHub repository. Models are not retrained. No original Polymarket source dataset is downloaded by the app. If the fine-tuned model repository is private, configure a read-only `HF_TOKEN` in Streamlit secrets/environment or publish the model publicly.

### Fine-tuned model configuration and the reported missing-cutoffs error

The deployed classifier is loaded directly from the **fine-tuned**, not the base, Hugging Face repository:

[2610gnam05/polypulse-deberta-v3-base-demand](https://huggingface.co/2610gnam05/polypulse-deberta-v3-base-demand)

The repository uses the standard Hugging Face **`config.json`** (provided by the user), with `id2label` mapping `0=Low`, `1=Medium`, `2=High` and `max_position_embeddings=512`. The original application incorrectly requested an additional `polypulse_config.json` and treated historical USD cutoffs as required, causing **every demand classification to fail**. This has been fixed:

- The classifier pipeline loads the fine-tuned model and tokenizer, then reads `classifier.model.config.to_dict()`; no additional config file is needed.
- If no saved validation adjustments appear in that configuration, classification uses the **raw** three softmax scores. The interface explicitly labels them as raw and does **not** claim to match any validation-tuned Colab predictions.
- If no actual historical USD classification thresholds are published, the category volume band is omitted. Historical **observed** comparable-market USD volumes and their quartiles are still shown by MiniLM. No fake cutoffs are computed.
- The effective sequence limit is chosen from the available published input/tokenizer/model-position limits. The included `max_position_embeddings=512` is an architectural limit, **not** proof that the training notebook used 512-token truncation.

These changes permit ordinary market questions to be classified, without manufacturing missing research metadata. For exact parity with **validation-adjusted** Google Colab outputs, the actual adjustment values and training-time input limit must be established from the Colab notebook; they cannot be reverse-engineered from the supplied `config.json`.

### Input/reset behavior

Inputs are ordinary Streamlit widgets rather than a `st.form`. Editing the market question or event title **does not** remove the previous analysis; it stays visible while the user prepares the next question. Only clicking **Analyse Market** removes the previous result before running inference and displaying the new analysis. The **Clear** button explicitly empties both input fields and the analysis. Streamlit text widgets commit on focus change or Enter, but these commits do not erase results.

## Popular Questions tab

The **🔥 Popular Questions** tab appears directly beside Market Analysis and draws from up to twelve historical example questions previously identified in the supplied Polymarket index. It checks that each example actually exists in `historical_markets.parquet` and has at least five markets meeting the index's similarity threshold before showing it. Categories include ₿ Crypto, ⚽ Football, 🏈 NFL, 🎮 Esports, 🗳️ Politics, 🎬 Entertainment, and 🤖 Technology.

The **Similar markets** count is computed with cosine similarity between the example's **stored MiniLM vector** and all reference vectors. The **Median historical volume** is computed from the ten nearest qualified markets (or fewer when fewer qualify), reading real `volume_usd` records from the parquet. No new model inference or original-dataset download is required merely to view this tab. A separate expander reveals each example's saved Event Title so users can reproduce its question-and-title query in Market Analysis.

**Important interpretation:** these examples are ranked by **coverage within a curated set**, not by the number of user requests or by top global Polymarket trading volume. There is no user-query-frequency data available. An example stored in the index also matches itself, so this tab is ideal for demonstrating index coverage but not for independent generalization claims. Some examples may disappear if the reference dataset changes.

## MiniLM history and compatibility

The corrected `data/metadata.json` declares:

- **20,000 records** in `historical_markets.parquet` and `market_embeddings.npy`.
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- **Question template**: `Question: {q} | Event: {title}` (title is optional).
- **Pooling**: masked mean over token states; **normalization**: L2.
- **Maximum tokens**: 256; **dimension**: 384.
- **Default similarity threshold**: 0.58; **minimum comparable markets**: 5.

These transformations must **not** be replaced with the DeBERTa text template or another embedding method. The supplied `historical_search.py` explicitly pins the pretrained MiniLM revision from metadata. Historical volumes always come from the corrected stored parquet data; the app does not estimate USD volumes using an LLM.

**Dataset caveats:** `split_exclusion_verified=false` in the supplied metadata; it was not proven that training/validation/test records were excluded from the 20,000-market reference set. Review this before claiming independent test accuracy. The metadata also warns that `end_date` is not proven to be an actual resolution timestamp and that volumes are not necessarily reconstructable at older historical cutoffs. This is descriptive reference analysis, not a controlled, leakage-free retrospective performance study.

## Testing

### Local unit tests (no Hugging Face download)

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py demand_prediction.py historical_search.py
```

Unit tests check input formatting, mock Low / Medium / High classification scores over **three different domains** (crypto, politics, sports), score summation, missing-optional-config inference, saved adjustment math, historical cutoffs, invalid/blank input, cosine ranking/volume attribution, insufficient comparables, and integrity of the 20,000-row embedding archive. **Mock outputs are synthetic test fixtures**, not reported model outputs or accuracy evidence.

### Live end-to-end tests (required before final academic submission)

With access to the published model repo and internet:

```bash
POLYPULSE_LIVE_TESTS=1 python -m unittest discover -s tests -v
```

On Windows PowerShell:

```powershell
$env:POLYPULSE_LIVE_TESTS="1"
python -m unittest discover -s tests -v
```

The live tests run three market questions (crypto, politics, sports) through **both real pipelines**, validate that scores sum to approximately 1, verify the archived USD volume for each returned market, and check that missing comparables do not lead to fabricated statistics. Record actual screenshots and inference timings from Streamlit Cloud as required by the class. **For definitive tuned-classification parity, also compare predictions and adjusted scores against the Colab notebook's held-out samples**; that notebook was not included in these attachments.

## Final model comparison results

`results/model_comparison.csv` contains rows for the three trained alternatives: **ModernBERT-base**, **Ettin-Encoder-150M**, **DeBERTa-v3-base**. Metrics are intentionally blank; the notebooks' real training dataset, counts, accuracy, macro-F1, training duration and confusion matrix have not been provided. Fill them from the final experiment run **without inventing numbers**. The Model Performance tab displays populated results once the file is edited.

Note that the dataset supporting the **historical similarity index** is a different artifact from whatever exact fine-tuning train/validation/test splits were used for DeBERTa. Do not infer the latter from historical metadata.

## GitHub and Streamlit Community Cloud

1. Create a new GitHub repository for PolyPulse, commit the contents of **this folder** (including `data/`; each bundled file is below GitHub's usual per-file size limit), and push to GitHub. Do not commit model weights or credentials.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/), create a new app and connect the repo, branch, and main file **`app.py`**. Select Python 3.11 if the runtime setting is available.
3. Make sure `config.json`, model weights and tokenizer are published in the indicated Hugging Face repository. If private, add a read-only `HF_TOKEN` via Streamlit app secrets.
4. Wait for first-run model download; the app then reuses cached pipelines. Observe available RAM: the CPU DeBERTa model is relatively large, so free-tier memory limits may still cause model loading failures. If so, evaluate the Community Cloud memory budget and a suitable upgraded host; do not change models or silently simplify scoring just to hide the limitation.
5. Run the three live test questions, check all metrics/volumes and take screenshots. If the notebook used adjustments, obtain them before asserting matching tuned predictions.

### Academic deliverables still to supply separately

The ISOM5240 project syllabus also calls for a PDF project report (<10 pages), Colab fine-tuning/testing notebooks with saved outputs, necessary fine-tuned model files, a PowerPoint presentation and MP4 video (<=10 minutes), experimental results in Excel, a public Streamlit Cloud URL, and a GitHub URL. **This ZIP contains the Streamlit business application only**, not the other separately graded deliverables. When making the final full academic submission, verify that the hosted model and submitted fine-tuned model are exactly the same.

## Additional notes

- Both pipelines use Hugging Face pre-trained models. One is fine-tuned by the project, the other is used as a pretrained feature extractor. The app does **not** introduce a third model or paid API.
- The archived 20,000 market volumes and embedding vectors must match; the loader raises an error for mismatches.
- Application code aims for simplicity and readable, business-school-friendly Python. Error details are available in UI expanders rather than silently substituted predictions.
- Measured times exclude initial model download/loading and vary based on deployment resource limits.
