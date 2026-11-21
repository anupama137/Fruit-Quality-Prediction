# Fruit Quality Prediction

A Flask app that predicts fruit quality from firmness, color score, sugar level, spots percentage, and days after harvest using a trained decision tree model.

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

## Deploy

The included `render.yaml` configures a Render web service using Gunicorn. Connect the GitHub repository to Render or create a Blueprint from the repository.
