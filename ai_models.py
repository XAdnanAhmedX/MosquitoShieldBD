
from typing import Tuple, Dict, Any, Union, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from data_generator import MEDICINES

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:

    features = ["Temperature", "Rainfall", "Humidity"]
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[features])
    return scaled_array, scaler

def train_outbreak_model(df: pd.DataFrame, scaler: StandardScaler) -> LogisticRegression:
    features = ["Temperature", "Rainfall", "Humidity"]
    X = scaler.transform(df[features])
    y = (df["Actual_Cases"] > 50).astype(int)

    if len(np.unique(y)) < 2:
        y[0] = 1

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model

def train_symptom_model(symptom_df: pd.DataFrame) -> LogisticRegression:
    features = ["Fever", "JointPain", "Rash", "Headache", "EyePain"]
    X = symptom_df[features]
    y = symptom_df["Has_Disease"]

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model


def predict_symptom_risk(
    symptoms: Dict[str, int],
    zone_risk: str,
    symptom_model: LogisticRegression
) -> Tuple[float, str]:

    feature_cols = ["Fever", "JointPain", "Rash", "Headache", "EyePain"]
    X = pd.DataFrame([[symptoms[c] for c in feature_cols]], columns=feature_cols)
    ml_prob = float(symptom_model.predict_proba(X)[0][1])

    if symptoms["Fever"] == 0:
        rule_prob = 0.02
        rule_note = "No fever — acute mosquito-borne illness is very unlikely."
    elif symptoms["Fever"] == 1 and symptoms["JointPain"] == 1 and symptoms["EyePain"] == 1:
        rule_prob = 0.90
        rule_note = "Classic dengue triad: fever + joint pain + eye pain."
    else:
        secondary = sum(symptoms[k] for k in ["JointPain", "Rash", "Headache", "EyePain"])
        rule_prob = min(0.72, 0.20 + 0.11 * secondary)
        rule_note = f"Fever with {secondary} additional symptom(s) — moderate baseline risk."

    zone_bump = {"High": 0.06, "Medium": 0.03, "Low": 0.0}.get(zone_risk, 0.0)

    if symptoms["Fever"] == 0:
        final = min(ml_prob, rule_prob + zone_bump)
    elif rule_prob >= 0.85:
        final = max(ml_prob, rule_prob) + zone_bump
    else:
        final = 0.55 * rule_prob + 0.45 * ml_prob + zone_bump

    final = float(min(0.98, max(0.01, final)))
    return final, rule_note
