# UI

This folder contains the Streamlit interface for Tasks 1-5.

## Run

From project root:

```powershell
./venv/Scripts/python.exe -m streamlit run UI/app.py
```

Or with default Python:

```powershell
streamlit run UI/app.py
```

## Persistence / Reuse

- Task outputs are reused from `Task*/output/` when already available.
- UI-level manifests are stored in `UI/model_cache/manifests/` to detect reusable runs.
- Task5 saves model checkpoints in `Task5/output/model_cache/*.pt` and a run manifest in `Task5/output/model_cache_manifest.json`.
- Closing and reopening Streamlit does not remove saved model files.
