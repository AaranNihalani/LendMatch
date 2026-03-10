import os
from datetime import datetime

import numpy as np
import pandas as pd


def _choice(rng, values, p=None, size=1):
    if p is not None:
        p = np.asarray(p, dtype=float)
        p = p / p.sum()
    return rng.choice(values, size=size, replace=True, p=p)


def _random_month_year(rng, start_year=1980, end_year=2018):
    year = int(rng.integers(start_year, end_year + 1))
    month = int(rng.integers(1, 13))
    dt = datetime(year, month, 1)
    return dt.strftime("%b-%Y")


def generate_synthetic_lendingclub_like_data(output_dir="data/raw", n_accepted=12000, n_rejected=8000, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    states = [
        "CA",
        "NY",
        "TX",
        "FL",
        "IL",
        "WA",
        "MA",
        "NJ",
        "GA",
        "VA",
        "PA",
        "NC",
        "OH",
        "MI",
        "AZ",
        "CO",
    ]
    purposes = [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "small_business",
        "major_purchase",
        "medical",
        "vacation",
        "moving",
        "car",
        "other",
    ]
    home_ownership = ["RENT", "MORTGAGE", "OWN", "OTHER"]
    verification_status = ["Verified", "Source Verified", "Not Verified"]
    initial_list_status = ["w", "f"]
    application_type = ["Individual", "Joint App"]
    emp_length = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]
    terms = [" 36 months", " 60 months"]

    def make_base(n):
        fico = rng.normal(loc=690, scale=45, size=n).clip(560, 850)
        loan_amnt = rng.lognormal(mean=np.log(12000), sigma=0.55, size=n).clip(1000, 40000)
        annual_inc = rng.lognormal(mean=np.log(65000), sigma=0.6, size=n).clip(12000, 250000)
        dti = rng.normal(loc=16, scale=7, size=n).clip(0, 45)
        revol_bal = rng.lognormal(mean=np.log(9000), sigma=0.9, size=n).clip(0, 120000)
        revol_util = rng.normal(loc=48, scale=18, size=n).clip(0, 130)
        total_acc = rng.normal(loc=24, scale=9, size=n).clip(1, 90).round()
        open_acc = (total_acc * rng.uniform(0.35, 0.75, size=n)).clip(1, 60).round()
        delinq_2yrs = rng.poisson(lam=0.4, size=n).clip(0, 15)
        inq_last_6mths = rng.poisson(lam=0.8, size=n).clip(0, 15)
        pub_rec = rng.poisson(lam=0.15, size=n).clip(0, 10)
        mort_acc = rng.poisson(lam=1.2, size=n).clip(0, 20)
        pub_rec_bankruptcies = rng.poisson(lam=0.08, size=n).clip(0, 10)
        tax_liens = rng.poisson(lam=0.02, size=n).clip(0, 10)

        term = _choice(rng, terms, p=[0.78, 0.22], size=n)
        term_months = np.where(pd.Series(term).str.contains("60"), 60.0, 36.0)

        addr_state = _choice(rng, states, size=n)
        purpose = _choice(rng, purposes, p=[0.44, 0.16, 0.08, 0.05, 0.06, 0.04, 0.03, 0.02, 0.04, 0.06], size=n)
        home = _choice(rng, home_ownership, p=[0.43, 0.45, 0.10, 0.02], size=n)
        ver = _choice(rng, verification_status, p=[0.35, 0.30, 0.35], size=n)
        ils = _choice(rng, initial_list_status, p=[0.82, 0.18], size=n)
        app_type = _choice(rng, application_type, p=[0.93, 0.07], size=n)
        emp = _choice(rng, emp_length, size=n)
        earliest_cr_line = [_random_month_year(rng, 1978, 2016) for _ in range(n)]

        fico_low = (fico - rng.uniform(0, 15, size=n)).clip(560, 850).round()
        fico_high = (fico_low + rng.uniform(0, 20, size=n)).clip(560, 850).round()

        base = pd.DataFrame(
            {
                "loan_amnt": loan_amnt.round(0),
                "term": term,
                "term_months": term_months,
                "annual_inc": annual_inc.round(2),
                "dti": dti.round(2),
                "fico_range_low": fico_low,
                "fico_range_high": fico_high,
                "revol_bal": revol_bal.round(0),
                "revol_util": revol_util.round(2),
                "total_acc": total_acc,
                "open_acc": open_acc,
                "delinq_2yrs": delinq_2yrs,
                "inq_last_6mths": inq_last_6mths,
                "pub_rec": pub_rec,
                "collections_12_mths_ex_med": rng.poisson(lam=0.05, size=n).clip(0, 5),
                "acc_now_delinq": rng.poisson(lam=0.02, size=n).clip(0, 3),
                "tot_coll_amt": rng.lognormal(mean=np.log(120), sigma=1.1, size=n).clip(0, 25000).round(0),
                "tot_cur_bal": rng.lognormal(mean=np.log(85000), sigma=0.8, size=n).clip(0, 500000).round(0),
                "total_rev_hi_lim": (revol_bal * rng.uniform(1.2, 2.8, size=n)).clip(0, 350000).round(0),
                "mort_acc": mort_acc,
                "pub_rec_bankruptcies": pub_rec_bankruptcies,
                "tax_liens": tax_liens,
                "home_ownership": home,
                "verification_status": ver,
                "purpose": purpose,
                "addr_state": addr_state,
                "initial_list_status": ils,
                "application_type": app_type,
                "emp_length": emp,
                "earliest_cr_line": earliest_cr_line,
            }
        )
        return base

    accepted = make_base(n_accepted)
    fico_mid = (accepted["fico_range_low"] + accepted["fico_range_high"]) / 2
    risk_term = np.where(accepted["term_months"] > 36, 1.1, 1.0)
    risk_score = (
        0.035 * (accepted["dti"].astype(float))
        + 0.12 * accepted["loan_amnt"].astype(float) / accepted["annual_inc"].astype(float).replace(0, 1)
        + 0.25 * (accepted["revol_util"].astype(float) / 100)
        + 0.10 * accepted["delinq_2yrs"].astype(float)
        + 0.06 * accepted["inq_last_6mths"].astype(float)
        + 0.05 * accepted["pub_rec"].astype(float)
        + 0.45 * (1 - (fico_mid - 560) / (850 - 560))
    ) * risk_term
    base_rate = 7.5 + 18.0 * risk_score.clip(0, 1.2) + rng.normal(0, 1.4, size=n_accepted)
    accepted["int_rate"] = base_rate.clip(5.0, 30.0).round(2)

    principal = accepted["loan_amnt"].astype(float).to_numpy()
    term_months = accepted["term_months"].astype(int).to_numpy()
    r = (accepted["int_rate"].astype(float).to_numpy() / 100.0) / 12.0
    payment = principal * (r * (1 + r) ** term_months) / ((1 + r) ** term_months - 1)
    accepted["installment"] = np.nan_to_num(payment, nan=principal / term_months).round(2)

    prob_default = (0.06 + 0.65 * risk_score).clip(0.02, 0.45)
    is_default = rng.random(n_accepted) < prob_default
    accepted["loan_status"] = np.where(is_default, "Charged Off", "Fully Paid")

    accepted["policy_code"] = 1

    rejected = pd.DataFrame(
        {
            "Amount Requested": make_base(n_rejected)["loan_amnt"].astype(float).round(0),
            "Risk_Score": rng.normal(loc=685, scale=55, size=n_rejected).clip(560, 850).round(0),
            "Debt-To-Income Ratio": (rng.normal(loc=22, scale=9, size=n_rejected).clip(0, 60)).round(2).astype(str),
            "State": _choice(rng, states, size=n_rejected),
            "Employment Length": _choice(rng, emp_length, size=n_rejected),
            "Policy Code": 0,
        }
    )
    rejected["Debt-To-Income Ratio"] = rejected["Debt-To-Income Ratio"].astype(str) + "%"

    accepted_path = os.path.join(output_dir, "accepted_synthetic.csv")
    rejected_path = os.path.join(output_dir, "rejected_synthetic.csv")
    accepted.to_csv(accepted_path, index=False)
    rejected.to_csv(rejected_path, index=False)

    return {"accepted": accepted_path, "rejected": rejected_path}


if __name__ == "__main__":
    paths = generate_synthetic_lendingclub_like_data()
    print(paths)
