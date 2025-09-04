#Using Faker library to generate fake data
import pandas as pd
from faker import Faker
import random
import numpy as np
from scipy.stats import truncnorm


fake = Faker('en_US')

#number of rows to generate
n = 8000

#generate fake data

data = {
    "patient": [f"Patient_{i+1}" for i in range(n)],  # Index-like patient names
    "age": [random.randint(18, 45) for _ in range(n)],
    "blood_group": [random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]) for _ in range(n)],
    "glucose": [round(random.uniform(70, 180), 1) for _ in range(n)],
    "insulin": [round(random.uniform(2, 25), 1) for _ in range(n)],
    "bmi": [round(random.uniform(15, 40), 1) for _ in range(n)],
    "cholesterol": [round(random.uniform(120, 280), 1) for _ in range(n)],
    "systolic_bp": [random.randint(90, 180) for _ in range(n)],
    "diastolic_bp": [random.randint(60, 120) for _ in range(n)],
    "heart_rate": [random.randint(50, 120) for _ in range(n)],
    "pregnancies": [random.randint(0, 10) for _ in range(n)],
    "first_pregnancy": [random.choice([0, 1]) for _ in range(n)],
    "earlier_birth": [random.choice([0, 1]) for _ in range(n)],
    "smoker": [random.choice([0, 1]) for _ in range(n)],
    "alcohol_consumption": [random.choice(["None", "Low", "Medium", "High"]) for _ in range(n)],
    "exercise_level": [random.choice(["Low", "Medium", "High"]) for _ in range(n)],
    "family_history_diabetes": [random.choice([0, 1]) for _ in range(n)],
    "allergies": [random.choice([0, 1]) for _ in range(n)],
    "medications": [random.choice([0, 1]) for _ in range(n)]
}

df = pd.DataFrame(data)

#add rule for pregancies and first pregnancy
df["first_pregnancy"] = [0 if preg != 0 else random.choice([0, 1]) for preg in df["pregnancies"]]


# Set patient as index
df.set_index("patient", inplace=True)

df.to_csv("patient_data.csv")


#making age follow gausian for machine learning tests

df_up=df.copy()

def truncated_normal(size, mean=30, sd=7, low=18, high=45):
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

ages = truncated_normal(len(df), mean=30, sd=7, low=18, high=45)
df_up["age"] = np.rint(ages).astype(int)

df_up.to_csv("patient_data_up.csv")



