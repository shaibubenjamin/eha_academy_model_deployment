# eHA Academy — Model Deployment (COVID-19 Risk Assessment)

A beginner-friendly example built for **Module 8B: Integrating AI into Model
Deployment**. It trains a small `scikit-learn` model on a synthetic COVID-19
symptoms dataset and serves it as an interactive [Streamlit](https://streamlit.io)
app.

> ⚠️ **Educational demo only.** `covid19_symptoms.csv` is synthetic
> (computer-generated) data, not real patient data. This is not a medical
> tool and must never be used for real diagnosis or treatment decisions.

## Files

| File | Purpose |
|---|---|
| `modeldeployment.py` | The Streamlit app — loads the CSV, trains the model, serves the UI. Heavily commented, line by line. |
| `requirements.txt` | Python dependencies. |
| `covid19_symptoms.csv` | Synthetic training data (300 rows). |

## Run it locally

```bash
pip install -r requirements.txt
streamlit run modeldeployment.py
```

Then open http://localhost:8501.

## Run it with Docker

The full `Dockerfile` content and the exact `docker build` / `docker run`
commands are documented in the comment block at the top of
`modeldeployment.py` — copy that block into a file named `Dockerfile` in
this folder, then:

```bash
docker build -t covid-risk-app:v1 .
docker run -p 8501:8501 covid-risk-app:v1
```

## About the model

Logistic regression trained on `covid19_symptoms.csv` (age + five yes/no
symptom fields) predicting Low Risk vs. High Risk, with training accuracy
shown live in the app. See the Docker/Streamlit slide deck for this
module for the full "why" behind each step.
