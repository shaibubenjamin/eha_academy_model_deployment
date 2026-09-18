"""
=============================================================================
 modeldeployment.py
 A beginner-friendly COVID-19 Risk Assessment app — built to match
 Module 8B: "Integrating AI into Model Deployment" (eHA Academy).
=============================================================================

WHAT THIS FILE DOES
--------------------
This is a single, self-contained Streamlit application. On startup, it
reads a CSV file (covid19_symptoms.csv) into a pandas DataFrame, trains a
small machine-learning model on it (no external model.pkl file needed),
then wraps that model in an interactive web page where a user can enter
symptoms and instantly see a risk estimate.

This mirrors a REAL model-deployment workflow much more closely than
hard-coding numbers in the script: a data scientist hands you a CSV (or a
database query), and your job is to load it, train on it, and serve it.

The whole project is just THREE files, so the whole "Serve -> Containerize
-> Deploy" journey from the slide deck is easy to follow end to end:
    1. modeldeployment.py       <- this file, the Streamlit app
    2. requirements.txt         <- the Python packages it needs
    3. covid19_symptoms.csv     <- the training data it learns from

IMPORTANT DISCLAIMER
---------------------
This app is an EDUCATIONAL example only. covid19_symptoms.csv is a
SYNTHETIC (computer-generated, made-up) dataset created for this course —
it is NOT real patient data. This is NOT a real medical tool and must
never be used to make actual health decisions. In a real project, this
file would be replaced with properly collected, ethically sourced patient
data, and the model would be validated by medical professionals before
ever reaching a user.


HOW TO RUN THIS LOCALLY (Step 1-3 from the slide deck)
--------------------------------------------------------
1. Save this file, requirements.txt, and covid19_symptoms.csv together
   in the same folder — the app will not start without the CSV, since
   that is what it trains on.
2. Create and activate a virtual environment (recommended, optional):
       python -m venv venv
       venv\\Scripts\\activate        (Windows)
       source venv/bin/activate       (macOS/Linux)
3. Install the dependencies listed in requirements.txt:
       pip install -r requirements.txt
4. Start the app:
       streamlit run modeldeployment.py
5. Streamlit will open your browser automatically at:
       http://localhost:8501


HOW TO CONTAINERIZE THIS WITH DOCKER (Step 4 from the slide deck)
---------------------------------------------------------------------
We are keeping this project to just a few plain files (no separate
"Dockerfile" delivered alongside them). Instead, copy the block below
into a new, extensionless file literally named "Dockerfile" in the same
folder as this script whenever you are ready to containerize it.

    # ---- copy everything between the lines into a file named "Dockerfile" ----
    FROM python:3.11-slim

    # Set the working directory inside the container
    WORKDIR /app

    # Copy only the requirements file first (this lets Docker cache this
    # step, so re-builds are fast when you only change app code later)
    COPY requirements.txt .

    # Install the Python dependencies inside the container
    RUN pip install --no-cache-dir -r requirements.txt

    # Now copy the rest of the project — the script AND the CSV it trains
    # on, since the app reads covid19_symptoms.csv on startup.
    COPY . .

    # Tell Docker which port Streamlit listens on
    EXPOSE 8501

    # The command that runs when the container starts
    CMD ["streamlit", "run", "modeldeployment.py", \\
         "--server.port=8501", "--server.address=0.0.0.0"]
    # ---- end of Dockerfile content ----

Then, from a terminal in that same folder, run exactly the three commands
taught on the Docker slide:

    $ docker build -t covid-risk-app:v1 .
    $ docker run -p 8501:8501 covid-risk-app:v1
    $ open http://localhost:8501          (or just visit it in your browser)

Because everything the container needs (Python, Streamlit, scikit-learn,
and this script) is baked into the image, this exact same set of commands
will work identically on your laptop, a teammate's laptop, or a cloud
host — that is the whole point of Docker, as covered on the
"Why Docker Matters" slide.
=============================================================================
"""

# -----------------------------------------------------------------------
# 1) IMPORTS
# -----------------------------------------------------------------------
# streamlit gives us the widgets (sliders, checkboxes, buttons) and the
# web page itself — we always import it as "st" by convention.
import streamlit as st

# numpy lets us build the small numeric array the model expects as input.
import numpy as np

# pandas is only used here to show the user a neat little summary table
# of what they entered before we predict on it.
import pandas as pd

# LogisticRegression is a simple, fast, easy-to-explain classifier —
# perfect for a first "Model Deployment" teaching example.
from sklearn.linear_model import LogisticRegression


# -----------------------------------------------------------------------
# 2) PAGE CONFIGURATION
# -----------------------------------------------------------------------
# st.set_page_config() must be the FIRST Streamlit command in the script.
# It controls the browser tab title/icon and the overall page layout.
st.set_page_config(
    page_title="COVID-19 Risk Assessment",   # text shown on the browser tab
    page_icon="🦠",                           # emoji shown on the browser tab
    layout="centered",                        # "centered" keeps content narrow and readable
)


# -----------------------------------------------------------------------
# 3) LOAD THE TRAINING DATA FROM CSV, THEN BUILD (AND CACHE) THE MODEL
#    — THE "GOLDEN RULE" FROM THE SLIDE DECK
# -----------------------------------------------------------------------
# The exact columns our CSV uses as model inputs, in the exact order the
# model expects them. Keeping this as one named list (instead of typing
# the column names out again later) means there is only one place to
# update if the CSV's columns ever change.
FEATURE_COLUMNS = [
    "age",
    "fever",
    "cough",
    "fatigue",
    "difficulty_breathing",
    "contact_with_case",
]

# The column in the CSV holding the "correct answer" we train on:
# 0 = Low Risk, 1 = High Risk.
LABEL_COLUMN = "risk"

# The CSV must live in the same folder as this script. Docker's
# "COPY . ." step (see the Dockerfile block above) brings it into the
# container automatically, so this path works locally AND in Docker.
DATA_PATH = "covid19_symptoms.csv"


# @st.cache_resource tells Streamlit: "run this function ONCE, keep the
# result (our trained model) in memory, and reuse it for every user and
# every interaction." Without this decorator, Streamlit would reload the
# CSV and retrain the model on every single click, which is slow and
# wasteful. This is exactly the Golden Rule from the "Serve Your Model
# With Streamlit" slide: load/build the model once, never inside a
# button click.
@st.cache_resource
def train_covid_risk_model():
    """
    Loads covid19_symptoms.csv, trains a logistic regression model on it,
    and returns the trained model along with two simple stats about it
    (how many rows it learned from, and how accurately it fits them) so
    the app can be transparent with the user about where it came from.

    In a REAL project, this function would more likely load an ALREADY
    trained model instead, e.g.:
        return joblib.load("model.pkl")
    We train fresh from a CSV here so this demo mirrors a realistic,
    beginner "load data -> train -> serve" workflow end to end.
    """

    # ---- 3a) Read the CSV file into a pandas DataFrame ----
    # Each row is one training example (one simulated patient); each
    # column is one piece of information about them. pandas automatically
    # reads the first line of the CSV as the column headers.
    data = pd.read_csv(DATA_PATH)

    # ---- 3b) Split the DataFrame into features (X) and label (y) ----
    # X = everything the model is allowed to look at when predicting.
    # data[FEATURE_COLUMNS] selects just those columns, in that order;
    # .values converts the result from a DataFrame into a plain numpy
    # array, which is the format scikit-learn expects.
    X_train = data[FEATURE_COLUMNS].values

    # y = the "correct answer" for each row, used only during training.
    y_train = data[LABEL_COLUMN].values

    # ---- 3c) Create and train the model ----
    # max_iter is raised from the default so training reliably converges;
    # random_state makes the result reproducible run to run.
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # ---- 3d) Measure how well the model fits its own training data ----
    # This is NOT a substitute for testing on unseen data (a real project
    # always holds out a separate test set!) — it's included here only as
    # a simple, honest sanity check we can show the user in the app.
    training_accuracy = model.score(X_train, y_train)

    # ---- 3e) Hand everything back to Streamlit to cache ----
    return model, training_accuracy, len(data)


# Call the cached function ONCE per app lifetime (thanks to @st.cache_resource
# above) and keep references to the trained model and its stats.
model, training_accuracy, training_rows = train_covid_risk_model()


# -----------------------------------------------------------------------
# 4) PAGE HEADER AND INTRODUCTION TEXT
# -----------------------------------------------------------------------
# st.title() renders one large heading at the top of the page.
st.title("🦠 COVID-19 Risk Assessment")

# st.markdown() renders formatted text; we use it here for a short intro.
st.markdown(
    "Answer a few quick questions and this demo app will estimate whether "
    "your symptom profile looks **Low Risk** or **High Risk**, using a "
    "small machine-learning model running live in this browser tab."
)

# st.warning() draws a highlighted yellow box — perfect for the disclaimer,
# so it cannot be missed by anyone using the app.
st.warning(
    "⚠️ **Educational demo only.** This is a teaching example for "
    "Module 8B and is trained on made-up data. It is **not** a medical "
    "device and must never be used for real diagnosis or treatment "
    "decisions. If you feel unwell, contact a qualified health provider."
)

# st.expander() creates a collapsible section — handy for extra detail
# that not every user needs to see immediately. Here we use it to show
# exactly where the model's "knowledge" came from, using the stats
# train_covid_risk_model() returned above.
with st.expander("ℹ️ About the model powering this app"):
    st.write(
        f"This model was trained on **{training_rows} rows** from "
        f"`covid19_symptoms.csv`, and correctly classifies "
        f"**{training_accuracy:.0%}** of that same training data. "
        "A real project would always test on data the model has never "
        "seen before — this simple check is shown here only to make the "
        "load-data-then-train workflow visible."
    )

# st.divider() just draws a thin horizontal line to visually separate
# the intro/disclaimer from the input form below it.
st.divider()


# -----------------------------------------------------------------------
# 5) INPUT WIDGETS — THIS IS THE "UI" PART STREAMLIT BUILDS FOR US
# -----------------------------------------------------------------------
# st.subheader() renders a smaller heading than st.title().
st.subheader("Tell us how you're feeling")

# st.slider() returns whatever integer the user drags the slider to.
# Arguments: label, minimum, maximum, default starting value.
age = st.slider("Age", min_value=0, max_value=100, value=30)

# st.columns(2) splits the page into two side-by-side areas, so the
# checkboxes below don't all stack in one long vertical list.
col1, col2 = st.columns(2)

# Each st.checkbox() returns True if ticked, False if not — with block
# lets us place the first pair of checkboxes in the left-hand column.
with col1:
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")

# ...and this "with" block places the second pair in the right-hand column.
with col2:
    fatigue = st.checkbox("Fatigue")
    difficulty_breathing = st.checkbox("Difficulty breathing")

# One more checkbox, spanning the full page width, for contact history.
contact_with_case = st.checkbox("Close contact with a confirmed case in the last 14 days")


# -----------------------------------------------------------------------
# 6) THE PREDICT BUTTON — WHERE THE MODEL ACTUALLY GETS USED
# -----------------------------------------------------------------------
# st.button() returns True only during the single rerun that happens right
# after the user clicks it — so everything indented under this "if" only
# runs at that moment.
if st.button("Assess My Risk", type="primary"):

    # ---- 6a) Turn the widget values into the exact array shape the ----
    # ---- model was trained on: [age, fever, cough, fatigue, ----
    # ----                        difficulty_breathing, contact] ----
    # int(True) is 1 and int(False) is 0, which is exactly the encoding
    # our training data above used for each symptom column.
    input_features = np.array([[
        age,
        int(fever),
        int(cough),
        int(fatigue),
        int(difficulty_breathing),
        int(contact_with_case),
    ]])

    # ---- 6b) Show the user a quick summary of what they entered, ----
    # ---- using a pandas DataFrame for a clean table display. ----
    summary = pd.DataFrame({
        "Field": ["Age", "Fever", "Cough", "Fatigue",
                  "Difficulty breathing", "Recent contact"],
        "Value": [age,
                  "Yes" if fever else "No",
                  "Yes" if cough else "No",
                  "Yes" if fatigue else "No",
                  "Yes" if difficulty_breathing else "No",
                  "Yes" if contact_with_case else "No"],
    })
    st.table(summary)

    # ---- 6c) Ask the trained model for a prediction. ----
    # model.predict() returns the predicted class: 0 (Low Risk) or 1 (High Risk).
    prediction = model.predict(input_features)[0]

    # model.predict_proba() returns the model's confidence for each class,
    # as [probability_of_0, probability_of_1]. We grab the High Risk (1) one.
    probability_high_risk = model.predict_proba(input_features)[0][1]

    # ---- 6d) Display the result, styled differently depending on risk. ----
    # st.divider() again, to separate the summary table from the result.
    st.divider()

    if prediction == 1:
        # st.error() draws a red box — used here to grab attention for
        # a High Risk result (red does NOT mean an actual error occurred).
        st.error(
            f"**Result: High Risk** "
            f"(model confidence: {probability_high_risk:.0%})\n\n"
            "Consider getting tested and consulting a health professional."
        )
    else:
        # st.success() draws a green box for a Low Risk result.
        st.success(
            f"**Result: Low Risk** "
            f"(model confidence: {(1 - probability_high_risk):.0%})\n\n"
            "Keep following good hygiene practices and monitor your symptoms."
        )

    # st.caption() renders small, muted text — a good spot for a final,
    # gentle reminder right next to the result itself.
    st.caption(
        "Reminder: this number comes from a tiny demo model trained on "
        "12 made-up examples. It is for learning purposes only."
    )


# -----------------------------------------------------------------------
# 7) FOOTER
# -----------------------------------------------------------------------
st.divider()
st.caption("Built for Module 8B: Integrating AI into Model Deployment — eHA Academy.")
