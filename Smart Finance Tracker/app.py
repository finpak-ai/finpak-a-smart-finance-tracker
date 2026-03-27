# app.py

import pandas as pd
import pickle
import os
import shutil
from datetime import date
from typing import List
from fastapi import FastAPI, Request, Form, Depends, status, Response, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import desc
from passlib.context import CryptContext

from src import config, processing, database, models

# --- Configuration ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
templates = Jinja2Templates(directory="templates")
models.Base.metadata.create_all(bind=database.engine)

# Currency Rates (Base: PKR)
# 1 PKR = X Target Currency
CURRENCY_RATES = {
    "PKR": 1.0,
    "INR": 0.30,   # 1 PKR approx 0.30 INR
    "USD": 0.0036  # 1 PKR approx 0.0036 USD (1 USD = 278 PKR)
}
CURRENCY_SYMBOLS = {"PKR": "Rs", "INR": "₹", "USD": "$"}

# --- Global Resources ---
models_ml = {}
pipeline = None
unique_occupations = ["Professional", "Retired", "Self_Employed", "Student"]
unique_cities = ["Tier_1", "Tier_2", "Tier_3"]

# --- Helper Functions ---
def get_user(request: Request, db: Session):
    user_id = request.cookies.get("user_id")
    if not user_id: return None
    return db.query(models.User).filter(models.User.id == user_id).first()

def convert_from_pkr(amount, currency):
    """Converts DB value (PKR) to User Preference"""
    if amount is None: return 0
    return round(amount * CURRENCY_RATES.get(currency, 1.0), 2)

def convert_to_pkr(amount, currency):
    """Converts User Input to DB value (PKR)"""
    if amount is None: return 0
    return amount / CURRENCY_RATES.get(currency, 1.0)

# --- Server Startup ---
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_ml, pipeline
    # Setup Static Dir
    if not os.path.exists("static"): os.makedirs("static")
    
    # Load ML
    raw_data = pd.read_csv(config.DATA_PATH)
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('label_encoder', processing.CustomLabelEncoder(cols=config.CAT_COLS)),
        ('outlier_capper', processing.OutlierCapper(cols=config.NUM_COLS)),
        ('feature_engineer', processing.FeatureEngineer(weights=config.WEEKLY_WEIGHTS, expense_cols=config.EXPENSE_COLS))
    ])
    pipeline.fit(raw_data)

    try:
        models_ml['overspend'] = pickle.load(open("saved_models/Overspend_Random_Forest.pkl", "rb"))
        models_ml['savings'] = pickle.load(open("saved_models/Savings_Random_Forest.pkl", "rb"))
        models_ml['health'] = pickle.load(open("saved_models/Health_Decision_Tree.pkl", "rb"))
        models_ml['suggestions'] = pickle.load(open("saved_models/Suggestions_Multi_Tree.pkl", "rb"))
        print("Models Loaded.")
    except: print("WARNING: ML Models not found.")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- AUTH ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/register")
async def register(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(database.get_db)):
    if db.query(models.User).filter(models.User.username == username).first():
        return templates.TemplateResponse("login.html", {"request": request, "error": "Username taken"})
    new_user = models.User(username=username, hashed_password=pwd_context.hash(password))
    db.add(new_user)
    db.commit()
    return templates.TemplateResponse("login.html", {"request": request, "success": "Account created!"})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user or not pwd_context.verify(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    response = RedirectResponse(url="/check_flow", status_code=303)
    response.set_cookie(key="user_id", value=str(user.id), httponly=True)
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse("/")
    response.delete_cookie("user_id")
    return response

@app.get("/check_flow")
async def check_flow(request: Request, db: Session = Depends(database.get_db)):
    user = get_user(request, db)
    if not user: return RedirectResponse("/", status_code=303)
    record = db.query(models.FinancialRecord).filter(models.FinancialRecord.user_id == user.id).first()
    return RedirectResponse("/dashboard" if record else "/input?mode=new", status_code=303)

# --- PROFILE ROUTES (New) ---
@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request, db: Session = Depends(database.get_db)):
    user = get_user(request, db)
    if not user: return RedirectResponse("/")
    return templates.TemplateResponse("profile.html", {"request": request, "user": user})

@app.post("/update_profile")
async def update_profile(
    request: Request,
    db: Session = Depends(database.get_db),
    full_name: str = Form(...),
    country: str = Form(...),
    currency: str = Form(...),
    profile_pic: UploadFile = File(None)
):
    user = get_user(request, db)
    if not user: return RedirectResponse("/")
    
    # Update Text Fields
    user.full_name = full_name
    user.country = country
    user.currency = currency

    # Handle File Upload
    if profile_pic and profile_pic.filename:
        file_location = f"static/{user.id}_{profile_pic.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(profile_pic.file, file_object)
        user.profile_pic = f"/{file_location}"
    
    db.commit()
    return RedirectResponse("/profile", status_code=303)

# --- DASHBOARD (Updated for Currency) ---
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(database.get_db)):
    user = get_user(request, db)
    if not user: return RedirectResponse("/")
    
    record = db.query(models.FinancialRecord).filter(models.FinancialRecord.user_id == user.id).order_by(desc(models.FinancialRecord.date)).first()
    if not record: return RedirectResponse("/input?mode=new")

    # ML Pipeline (Uses PKR Data directly from DB)
    input_data = {
        'Income': [record.income], 'Age': [record.age], 'Dependents': [record.dependents],
        'Occupation': [record.occupation], 'City_Tier': [record.city_tier],
        'Desired_Savings': [record.desired_savings],
        'Rent': [record.rent], 'Groceries': [record.groceries], 'Transport': [record.transport],
        'Eating_Out': [record.eating_out], 'Entertainment': [record.entertainment],
        'Utilities': [record.utilities], 'Healthcare': [record.healthcare],
        'Education': [record.education], 'Miscellaneous': [record.miscellaneous],
        'Insurance': [record.insurance], 'Loan_Repayment': [record.loan_repayment]
    }
    df_processed = pipeline.transform(pd.DataFrame(input_data))

    # ML Predictions (PKR)
    pred_overspend = models_ml['overspend'].predict(df_processed[config.FEATURES_OVERSPEND].fillna(0))[0]
    pred_savings_pkr = models_ml['savings'].predict(df_processed[config.FEATURES_SAVINGS].fillna(0))[0]
    
    pred_health = models_ml['health'].predict(df_processed[config.FEATURES_HEALTH].fillna(0))[0]
    try: health_status = {0: "Good", 1: "Average", 2: "Poor"}.get(int(pred_health), str(pred_health))
    except: health_status = str(pred_health)

    pred_suggestions = models_ml['suggestions'].predict(df_processed[config.FEATURES_SUGGESTIONS].fillna(0))[0]
    
    # --- CONVERT TO USER CURRENCY ---
    curr = user.currency
    sym = CURRENCY_SYMBOLS.get(curr, "")
    
    # Convert Savings
    display_savings = convert_from_pkr(pred_savings_pkr, curr)
    
    # Convert Suggestions
    suggestions = []
    for name, value_pkr in zip(config.TARGETS_SUGGESTIONS, pred_suggestions):
        if value_pkr > 0:
            clean_name = name.replace("Potential_Savings_", "").replace("_", " ")
            suggestions.append({"category": clean_name, "amount": convert_from_pkr(value_pkr, curr)})

    # Convert Record for Chart
    display_record = {
        "rent": convert_from_pkr(record.rent, curr),
        "groceries": convert_from_pkr(record.groceries, curr),
        "transport": convert_from_pkr(record.transport, curr),
        "eating_out": convert_from_pkr(record.eating_out, curr),
        "entertainment": convert_from_pkr(record.entertainment, curr),
        "utilities": convert_from_pkr(record.utilities, curr),
        "healthcare": convert_from_pkr(record.healthcare, curr),
        "education": convert_from_pkr(record.education, curr),
        "miscellaneous": convert_from_pkr(record.miscellaneous, curr),
        "insurance": convert_from_pkr(record.insurance, curr),
        "loan_repayment": convert_from_pkr(record.loan_repayment, curr)
    }

    return templates.TemplateResponse("dashboard.html", {
        "request": request, "user": user, "overspend": "YES" if pred_overspend == 1 else "NO",
        "predicted_savings": display_savings, "health": health_status,
        "suggestions": suggestions, "record_date": record.date, "record": display_record,
        "symbol": sym
    })

# --- INPUT PAGE ---
@app.get("/input", response_class=HTMLResponse)
async def input_page(request: Request, mode: str = "new", db: Session = Depends(database.get_db)):
    user = get_user(request, db)
    if not user: return RedirectResponse("/")
    
    record_data = {}
    record_id = ""
    current_date = date.today()
    curr = user.currency
    sym = CURRENCY_SYMBOLS.get(curr, "")

    if mode == "edit":
        latest = db.query(models.FinancialRecord).filter(models.FinancialRecord.user_id == user.id).order_by(desc(models.FinancialRecord.date)).first()
        if latest:
            record_id = latest.id; current_date = latest.date
            # Convert PKR DB values to User Currency for display
            record_data = {k: convert_from_pkr(getattr(latest, k), curr) for k in config.EXPENSE_COLS + ['income', 'desired_savings']}
            record_data['age'] = latest.age; record_data['dependents'] = latest.dependents
            record_data['occupation'] = latest.occupation; record_data['city_tier'] = latest.city_tier

    return templates.TemplateResponse("input.html", {
        "request": request, "user": user, "occupations": unique_occupations,
        "cities": unique_cities, "record": record_data, "record_id": record_id,
        "default_date": current_date, "mode": mode, "symbol": sym
    })

@app.post("/submit_data")
async def submit_data(
    request: Request, db: Session = Depends(database.get_db),
    record_id: str = Form(None), record_date: str = Form(...),
    income: float = Form(...), desired_savings: float = Form(...),
    rent: float = Form(...), groceries: float = Form(...), transport: float = Form(...),
    eating_out: float = Form(...), entertainment: float = Form(...), utilities: float = Form(...),
    healthcare: float = Form(...), education: float = Form(...), misc: float = Form(...),
    insurance: float = Form(...), loan: float = Form(...),
    age: int = Form(...), dependents: int = Form(...), occupation: str = Form(...), city: str = Form(...)
):
    user = get_user(request, db)
    if not user: return RedirectResponse("/")
    
    # CONVERT INPUT (User Currency) -> DB (PKR)
    curr = user.currency
    income = convert_to_pkr(income, curr); desired_savings = convert_to_pkr(desired_savings, curr)
    rent = convert_to_pkr(rent, curr); groceries = convert_to_pkr(groceries, curr)
    transport = convert_to_pkr(transport, curr); eating_out = convert_to_pkr(eating_out, curr)
    entertainment = convert_to_pkr(entertainment, curr); utilities = convert_to_pkr(utilities, curr)
    healthcare = convert_to_pkr(healthcare, curr); education = convert_to_pkr(education, curr)
    misc = convert_to_pkr(misc, curr); insurance = convert_to_pkr(insurance, curr)
    loan = convert_to_pkr(loan, curr)

    date_obj = date.fromisoformat(record_date)
    if record_id and record_id != "None" and record_id != "":
        rec = db.query(models.FinancialRecord).filter(models.FinancialRecord.id == record_id).first()
        if rec and rec.user_id == user.id:
            rec.date = date_obj; rec.income = income; rec.age = age; rec.dependents = dependents
            rec.occupation = occupation; rec.city_tier = city; rec.desired_savings = desired_savings
            rec.rent = rent; rec.groceries = groceries; rec.transport = transport
            rec.eating_out = eating_out; rec.entertainment = entertainment; rec.utilities = utilities
            rec.healthcare = healthcare; rec.education = education; rec.miscellaneous = misc
            rec.insurance = insurance; rec.loan_repayment = loan
    else:
        new_rec = models.FinancialRecord(
            user_id=user.id, date=date_obj, income=income, age=age, dependents=dependents,
            occupation=occupation, city_tier=city, desired_savings=desired_savings,
            rent=rent, groceries=groceries, transport=transport, eating_out=eating_out,
            entertainment=entertainment, utilities=utilities, healthcare=healthcare,
            education=education, miscellaneous=misc, insurance=insurance, loan_repayment=loan
        )
        db.add(new_rec)
    db.commit()
    return RedirectResponse("/dashboard", status_code=303)

# --- TRENDS & HISTORY (Updated for Currency) ---
@app.get("/history", response_class=HTMLResponse)
async def history(request: Request, db: Session = Depends(database.get_db)):
    user = get_user(request, db)
    if not user: return RedirectResponse("/")
    records = db.query(models.FinancialRecord).filter(models.FinancialRecord.user_id == user.id).order_by(desc(models.FinancialRecord.date)).all()
    
    # Process for display
    curr = user.currency; sym = CURRENCY_SYMBOLS.get(curr, "")
    display_records = []
    for r in records:
        total = sum([r.rent, r.groceries, r.transport, r.eating_out, r.entertainment, r.utilities, r.healthcare, r.education, r.miscellaneous, r.insurance, r.loan_repayment])
        display_records.append({
            "date": r.date, "occupation": r.occupation,
            "income": convert_from_pkr(r.income, curr),
            "total_expenses": convert_from_pkr(total, curr)
        })
    return templates.TemplateResponse("history.html", {"request": request, "user": user, "records": display_records, "symbol": sym})

@app.get("/trends", response_class=HTMLResponse)
async def trends(request: Request, db: Session = Depends(database.get_db)):
    user = get_user(request, db)
    if not user: return RedirectResponse("/")
    records = db.query(models.FinancialRecord).filter(models.FinancialRecord.user_id == user.id).order_by(models.FinancialRecord.date).all()
    
    curr = user.currency
    dates = [r.date.strftime("%b %Y") for r in records]
    
    def get_list(attr): return [convert_from_pkr(getattr(r, attr), curr) for r in records]
    
    total_exp = [sum([r.rent, r.groceries, r.transport, r.eating_out, r.entertainment, r.utilities, r.healthcare, r.education, r.miscellaneous, r.insurance, r.loan_repayment]) for r in records]
    
    data = {
        "dates": dates, "income": get_list("income"),
        "total_expenses": [convert_from_pkr(t, curr) for t in total_exp],
        "rent": get_list("rent"), "groceries": get_list("groceries"), "transport": get_list("transport"),
        "eating_out": get_list("eating_out"), "entertainment": get_list("entertainment"), "utilities": get_list("utilities"),
        "healthcare": get_list("healthcare"), "education": get_list("education"), "misc": get_list("miscellaneous"),
        "insurance": get_list("insurance"), "loan": get_list("loan_repayment")
    }
    return templates.TemplateResponse("trends.html", {"request": request, "user": user, "data": data, "symbol": CURRENCY_SYMBOLS.get(curr, "")})