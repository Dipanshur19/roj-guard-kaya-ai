# Applying the P0-fixed build to your existing VS Code project

## Recommended method: use the fixed ZIP as the new project folder

This is the lowest-risk approach because the model features, seed data, database and trained model artifacts changed together.

1. Stop the FastAPI and Streamlit terminals in VS Code.
2. Make a backup copy of your current project folder.
3. Extract `roj_guard_round2_p0_fixed.zip` into a new folder, for example `roj_guard_round2_p0_fixed`.
4. Copy only your private `.env` file from the old project into the new folder. Do not upload or commit it.
5. Create a new virtual environment and install `requirements.txt`.
6. Run `python seed_data_layer1.py`. This resets the demo database and regenerates current-date synthetic data and trained models.
7. Start `uvicorn main_layer1:app --reload`.
8. In another VS Code terminal start `streamlit run dashboard_layer3.py`.
9. Open `http://localhost:8501`.

The ZIP already contains a seeded database and model artifacts, but running the seed script once is recommended because all live ROJ dates are generated relative to the day you run it.

## If you want to patch your existing folder in place

Back up the folder first, preserve your `.env`, then replace/add these files from the fixed ZIP:

```text
main_layer1.py
feature_engineering_layer2.py
feature_engineering_layer3.py
train_models_layer3.py
inference_layer3.py
api_layer3.py
graph_builder_layer2.py
sync_service_layer2.py
models_layer4.py
agents_layer4.py
api_layer4.py
dashboard_layer3.py
seed_data_layer1.py
execution_layer4.py       NEW
time_utils.py              NEW
.env.example
.gitignore                 NEW/UPDATED
README.md
```

Then run:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python seed_data_layer1.py
uvicorn main_layer1:app --reload
```

Open another terminal:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run dashboard_layer3.py
```

## Do not carry these from the old ZIP into a submission

```text
.env
venv/
__pycache__/
```

Your old `.env` can remain locally for development, but it should never be in the submission ZIP or Git repository.

## Database note

`seed_data_layer1.py` intentionally resets `roj_guard.db`. The previous demo database was dominated by already-delivered lines and did not support the corrected live-feature model. If you have non-demo data you need to preserve, copy the old database somewhere safe before running the seed script.
