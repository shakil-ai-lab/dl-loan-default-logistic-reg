import streamlit as st
import pandas as pd
import joblib
import os


# Always load the file from the same folder as app.py
model_path = os.path.join(os.path.dirname(__file__), "final_pipeline_with_meta.pkl")
model = joblib.load(model_path)
print(model["model"])
