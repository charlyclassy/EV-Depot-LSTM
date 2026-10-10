# Optivolt EV Depot Intelligence Portal — LSTM matched-UI version

This version keeps the visual design and detailed plotting workflow of the
previous XGBoost Streamlit app, while adapting the selected model to LSTM.

Important:
- Deploy with Python 3.12 on Streamlit Community Cloud.
- Keep your existing nine UK Power Networks Optimise Prime depot CSV files
  in the repository root.
- Upload the files in this package to the same repository root.
- `app.py` replaces the previous app.py.
- `requirements.txt` replaces the previous requirements file.

Selected model: LSTM
Comparison: previous-day same-time baseline, Random Forest, XGBoost, LSTM.
Planning Capacity Threshold terminology is retained.
