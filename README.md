# Optivolt EV Depot Intelligence Portal — LSTM selected model

Run locally:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app preserves the multi-depot, upload, scenario, performance, alerts, reports,
archive and admin workflow while changing the selected operational model to LSTM.

Final EPA comparison:
- Previous-day same-time baseline
- Random Forest
- XGBoost
- LSTM (selected final model)

The included Bexleyheath evidence CSVs allow the app to display the final EPA
held-out results and recursive 24-hour LSTM forecast without retraining during
normal Bexleyheath navigation.

Planning thresholds are planning values, not verified contracted grid limits.
The bundled Optimise Prime data are historical and do not represent live telemetry.
