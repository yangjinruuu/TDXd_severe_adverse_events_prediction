# TDXd–Related Severe Adverse Events (Grade ≥3) Prediction Model

This folder contains the deployable Streamlit application for predicting the risk of TDXd-related severe adverse events (grade ≥3; `AE3_All`).

## Local deployment

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Application features

- Single-patient risk prediction using eight clinical variables.
- The interface displays only the final ensemble probability.
- ER is displayed as `Yes`/`No` in the interface and remains internally encoded as `1`/`0` for model prediction.
- Risk stratification:
  - Low risk: <15%
  - Intermediate risk: 15% to the internal Youden threshold (20.5%)
  - High risk: ≥ the internal Youden threshold (20.5%)
- Batch prediction from a CSV file.
- Global and local explanation figures bundled in the `assets/` directory.

## Required repository structure

```text
app.py
requirements.txt
runtime.txt
data/
  train_data_ae3_all.csv
  sample_batch_template.csv
assets/
  global/
  local/
  pdp/
README.md
```

## GitHub and Streamlit Community Cloud

1. Upload the contents of this folder to a GitHub repository.
2. Confirm that `app.py`, `data/`, and `assets/` are in the repository root.
3. In Streamlit Community Cloud, select the correct GitHub repository and branch.
4. Set the main file path to `app.py`.
5. Deploy the application.

After updating the repository, commit and push all changed files. If the app does not refresh automatically, use **Reboot app** or clear the Streamlit cache.

## Important note

This application is a clinical decision-support tool and does not replace clinical judgment. Treatment decisions should consider the patient's clinical status, drug labeling, guidelines, and multidisciplinary review.
