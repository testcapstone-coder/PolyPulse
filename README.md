# PolyPulse

**AI-Powered Prediction Market Intelligence** — HKUST ISOM5240 Deep Learning business application.

PolyPulse helps users evaluate the likely *demand category* (Low / Medium / High) for a proposed Polymarket question and investigate observed trading volumes from 20,000 similar historical markets. It **does not** predict the event outcome, guarantee future trading volume, or make trading/launch recommendations.

## Features

- **Demand:** Fine-tuned DeBERTa-v3-base Hugging Face `text-classification` pipeline, CPU-only. The exact text contract is `Question: {question.strip()} [SEP] Event: {event_title.strip()}`. It retrieves **training-time label mappings, saved validation decision adjustments, volume cutoffs, and sequence length** from the model owner's published `polypulse_config.json`; there are no hardcoded trained thresholds or invented settings in this app.
- **Comparables:** Pretrained all-MiniLM-L6-v2 Hugging Face `feature-extraction` pipeline, CPU-only. Reuses the supplied `historical_search.py`, the existing 384-dimensional vectors, and cosine similarity to find up to 10 historical markets and their *actual stored* trading volumes in USD.
- **Historical range:** Median and 25th/75th percentiles of the returned sufficiently similar markets only. These statistics disappear if fewer than five comparables meet the threshold.
- **Two tabs:** Market Analysis and Model Performance (experimental comparison metrics must be provided separately).
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

### Fine-tuned model configuration is a hard dependency

The required Hugging Face model repository is:

[2610gnam05/polypulse-deberta-v3-base-demand](https://huggingface.co/2610gnam05/polypulse-deberta-v3-base-demand)

`demand_prediction.py` calls `hf_hub_download(repo_id=..., filename="polypulse_config.json")` and loads the corresponding fine-tuned model **and tokenizer**, both from the same repository. It never falls back to `microsoft/deberta-v3-base`.

**Important validation limitation:** During packaging, the Hugging Face model repository and its published `polypulse_config.json` could not be reached from the development environment (internet resolution failed, and web access could not inspect this specific repository). Consequently, the exact configuration keys and prediction parity with the original Colab notebook have **not** been verified. Rather than fabricate thresholds/bias values, the code accepts a small set of explicit key representations; if it cannot interpret the published values safely, it raises a descriptive `ModelConfigurationError`. The historical MiniLM search branch still runs independently. Before claiming full DeBERTa parity, inspect the published file, compare the implementation with the original training notebook, and run live tests.

The expected configuration **semantics** (not concrete values) are:

| Required information | Notes |
|---|---|
| Demand label ordering | Model or published config must map the 3 indices to Low / Medium / High. |
| Maximum sequence length | Read from the published JSON, e.g., a `max_length` field. |
| Historical USD cutoffs | Two actual cutoffs that separate Low / Medium / High. No fabricated values. |
| Saved validation decision adjustments | Explicit method (logit bias, additive or multiplicative score) and saved per-class values. |

**Do not** substitute example numbers or remove adjustments to make the app run. If the config uses different keys or formulas, adapt `demand_prediction.py` to match the notebook exactly, then compare several specific predictions and classification scores to the Colab outputs. For logits, applying bias to `log(p)` is mathematically equivalent to adding the same bias to logits before softmax. Post-adjustment scores are shown as **model scores, not necessarily calibrated probabilities**.

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

Unit tests check input formatting, mock Low / Medium / High classification scores over **three different domains** (crypto, politics, sports), score summation, saved adjustment math, historical cutoffs, invalid/blank input, cosine ranking/volume attribution, insufficient comparables, and integrity of the 20,000-row embedding archive. **Mock outputs are synthetic test fixtures**, not reported model outputs or accuracy evidence.

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
3. Make sure `polypulse_config.json`, model weights and tokenizer are published in the indicated Hugging Face repository. If private, add a read-only `HF_TOKEN` via Streamlit app secrets.
4. Wait for first-run model download; the app then reuses cached pipelines. Observe available RAM: the CPU DeBERTa model is relatively large, so free-tier memory limits may still cause model loading failures. If so, evaluate the Community Cloud memory budget and a suitable upgraded host; do not change models or silently simplify scoring just to hide the limitation.
5. Run the three live test questions, check all metrics/volumes and take screenshots. Confirm model adjustments and outputs against the original Colab notebook.

### Academic deliverables still to supply separately

The ISOM5240 project syllabus also calls for a PDF project report (<10 pages), Colab fine-tuning/testing notebooks with saved outputs, necessary fine-tuned model files, a PowerPoint presentation and MP4 video (<=10 minutes), experimental results in Excel, a public Streamlit Cloud URL, and a GitHub URL. **This ZIP contains the Streamlit business application only**, not the other separately graded deliverables. When making the final full academic submission, verify that the hosted model and submitted fine-tuned model are exactly the same.

## Additional notes

- Both pipelines use Hugging Face pre-trained models. One is fine-tuned by the project, the other is used as a pretrained feature extractor. The app does **not** introduce a third model or paid API.
- The archived 20,000 market volumes and embedding vectors must match; the loader raises an error for mismatches.
- Application code aims for simplicity and readable, business-school-friendly Python. Error details are available in UI expanders rather than silently substituted predictions.
- Measured times exclude initial model download/loading and vary based on deployment resource limits.
