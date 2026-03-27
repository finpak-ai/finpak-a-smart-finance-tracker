# FinPak — A Smart Finance Tracker

A lightweight personal finance tracker built with **FastAPI**, **SQLite**, and **scikit-learn**. FinPak helps you record income & expenses, visualize spending trends, and get ML-powered insights such as overspending risk, predicted savings, financial health classification, and suggested areas to save.

> **Note:** This repository does **not** include the dataset file in the root. Follow the *Dataset setup* section below to download it from Kaggle and place it correctly.

---

## Features

- **User accounts**: register/login/logout
- **Profile**: name, country, preferred currency (PKR/INR/USD) and profile photo upload
- **Record finances**: income, desired savings, and common expense categories
- **Dashboard**:
  - Overspending prediction (Yes/No)
  - Predicted savings amount
  - Financial health (Good/Average/Poor)
  - Savings suggestions by category
  - Expense breakdown charts
- **History & trends**: view historical records and spending/income trends over time
- **ML pipeline**: preprocessing + feature engineering

---

## Tech stack

- Backend: **FastAPI**
- Templates/UI: **Jinja2** + HTML
- Database: **SQLite** (SQLAlchemy ORM)
- ML: **scikit-learn** (pickled models in `Smart Finance Tracker/saved_models/`)

---

## Project structure

```
finpak-a-smart-finance-tracker/
├── README.md
├── LICENSE
└── Smart Finance Tracker/
    ├── app.py
    ├── main.py
    ├── requirements.txt
    ├── finance.db
    ├── saved_models/
    ├── src/
    ├── templates/
    └── static/
```

---

## Dataset setup (required)

The app expects a CSV called **`data.csv`** (see `Smart Finance Tracker/src/config.py`).

1. Download the dataset from Kaggle:
   - `Indian Personal Finance and Spending Habits`
   - Link: https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits
2. Extract the downloaded archive.
3. Copy the main CSV file and rename it to **`data.csv`**.
4. Place **`data.csv` in the `Smart Finance Tracker/` directory** (same folder as `app.py`).

### Expected path

When you run the app from inside `Smart Finance Tracker/`, the dataset should be found at:

```
Smart Finance Tracker/data.csv
```

---

## Setup & run (local)

### 1) Clone

```bash
git clone https://github.com/finpak-ai/finpak-a-smart-finance-tracker.git
cd finpak-a-smart-finance-tracker
```

### 2) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies

```bash
cd "Smart Finance Tracker"
pip install -r requirements.txt
```

### 4) Add the dataset

Place `data.csv` as described in **Dataset setup**.

### 5) Start the web app

```bash
uvicorn app:app --reload
```

Open:
- http://127.0.0.1:8000

---

## ML models

Pre-trained models are stored in:

- `Smart Finance Tracker/saved_models/`

At startup, the app attempts to load:

- `Overspend_Random_Forest.pkl`
- `Savings_Random_Forest.pkl`
- `Health_Decision_Tree.pkl`
- `Suggestions_Multi_Tree.pkl`

### Important note about removed Savings model

`saved_models/Savings_Random_Forest.pkl` has been removed from this GitHub repo due to its large file size.

If you want **Savings prediction** to work:
- Pick **any one** Savings model file (for example `Savings_Linear_Regression.pkl` or `Savings_Decision_Tree.pkl`) from the `saved_models/` folder.
- Check the **notebook results/performance** (R², MAE, etc.) and choose the one with the best performance.
- Then **paste/rename** it to match what the app expects:

```
Smart Finance Tracker/saved_models/Savings_Random_Forest.pkl
```

> In short: choose the best-performing Savings model from your trained models and place it in `saved_models/` using the filename `Savings_Random_Forest.pkl` so the app can load it.

---

## (Optional) Train / regenerate models

If you want to retrain models locally (and recreate any missing `.pkl` files):

```bash
cd "Smart Finance Tracker"
python main.py
```

This script reads `data.csv`, runs preprocessing/feature engineering, trains multiple models for each task, evaluates them, and saves the resulting pickles into `saved_models/`.

---

## Security note

This project is intended for learning/demo purposes. Do not deploy as-is for production without adding proper secret management, session handling, CSRF protection
