# Optivolt EV Depot Intelligence Portal — saved-model deployment

Selecting a built-in depot now loads its saved depot-specific LSTM model.
Normal navigation does not train the model.

## One-time model build

1. Copy this package into the GitHub repository.
2. Keep the nine Optimise Prime depot CSV files in the repository root.
3. In GitHub, open **Actions**.
4. Select **Build saved LSTM depot models**.
5. Click **Run workflow**.
6. Wait for it to finish. The workflow commits the saved model files to `models/`.
7. Reboot the Streamlit app.

After that, Bexleyheath, Dartford, Islington, Mount Pleasant, Orpington,
Premier Park, Whitechapel, Camden, and Victoria load saved models and run
inference only. Training remains an explicit operation for uploaded data.
