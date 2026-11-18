"""Generate synthetic sample transactions (no real data). Run: python data/generate_sample_data.py"""
import csv
import random
from datetime import date
from pathlib import Path

random.seed(42)
rows = []


def add(d, desc, amt, typ):
    rows.append((d.isoformat(), desc, round(float(amt), 2), typ))


MONTHS = [(2026, 7), (2026, 8), (2026, 9), (2026, 10)]
FOOD_SMALL = ["Swiggy Order", "Zomato Order", "Starbucks Coffee", "Domino's Pizza", "Local Restaurant", "Chai Point"]
GROCERY = ["BigBasket Groceries", "DMart Groceries", "Blinkit Groceries"]
RIDES = ["Uber Ride", "Ola Cab", "Chennai Metro Card Recharge", "Petrol - Indian Oil", "Rapido Bike Ride"]
SHOP = ["Amazon Purchase", "Flipkart Purchase", "Myntra Fashion", "Nykaa Order"]
FUN = ["BookMyShow Movie Tickets", "PVR Cinemas", "Steam Game Purchase"]
HEALTH = ["Apollo Pharmacy", "Dr. Ravi Clinic Consultation", "PharmEasy Medicines"]

for y, m in MONTHS:
    day = lambda: random.randint(1, 28)
    add(date(y, m, 1), "Salary - Acme Technologies", 85000, "Income")
    add(date(y, m, 2), "House Rent", 22000, "Expense")
    add(date(y, m, 5), "Electricity Bill - TNEB", random.randint(1400, 2600), "Expense")
    add(date(y, m, 6), "Airtel Broadband", 1179, "Expense")
    add(date(y, m, 7), "Jio Mobile Recharge", 399, "Expense")
    add(date(y, m, 10), "Netflix Subscription", 649, "Expense")
    add(date(y, m, 10), "Spotify Premium", 119, "Expense")
    for _ in range(random.randint(10, 14)):
        add(date(y, m, day()), random.choice(FOOD_SMALL), random.randint(150, 650), "Expense")
    for _ in range(random.randint(3, 4)):
        add(date(y, m, day()), random.choice(GROCERY), random.randint(900, 2400), "Expense")
    for _ in range(random.randint(6, 9)):
        add(date(y, m, day()), random.choice(RIDES), random.randint(90, 1800), "Expense")
    for _ in range(random.randint(2, 4)):
        add(date(y, m, day()), random.choice(SHOP), random.randint(700, 3500), "Expense")
    for _ in range(random.randint(1, 3)):
        add(date(y, m, day()), random.choice(FUN), random.randint(300, 1200), "Expense")
    for _ in range(random.randint(1, 2)):
        add(date(y, m, day()), random.choice(HEALTH), random.randint(250, 1400), "Expense")
    add(date(y, m, day()), "ATM Withdrawal", random.choice([2000, 3000, 5000]), "Expense")
    if m in (8, 10):
        add(date(y, m, 18), "Freelance Payment - Upwork", random.randint(12000, 18000), "Income")

# A couple of deliberately unusual transactions for the insights demo
add(date(2026, 9, 14), "Amazon - Laptop Stand and Monitor", 18999, "Expense")
add(date(2026, 10, 12), "Apollo Hospitals - Dental Procedure", 7800, "Expense")

rows.sort(key=lambda r: r[0])
out = Path(__file__).resolve().parent / "sample_transactions.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "description", "amount", "type"])
    w.writerows(rows)
print(f"Wrote {len(rows)} rows to {out}")
