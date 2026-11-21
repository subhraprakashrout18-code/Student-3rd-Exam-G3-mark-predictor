import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Required ONLY because IQRCapper is a custom class inside your saved pipeline
class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5): self.factor = factor
    def fit(self, X, y=None):
        Q1, Q3 = np.nanpercentile(X, [25, 75], axis=0)
        self.lower_, self.upper_ = Q1 - self.factor * (Q3 - Q1), Q3 + self.factor * (Q3 - Q1)
        return self
    def transform(self, X): return np.clip(np.asarray(X), self.lower_, self.upper_)

model = joblib.load('student_g3_pipeline.pkl')

st.title("🎓 Student G3 Predictor")

# 9 UI inputs for the 9 columns your pipeline expects
data = {
    'sex': st.selectbox("Sex", ["F", "M"]),
    'age': st.number_input("Age", 15, 22, 17),
    'studytime': st.selectbox("Study Time (1-4)", [1, 2, 3, 4]),
    'failures': st.selectbox("Failures", [0, 1, 2, 3]),
    'schoolsup': st.selectbox("School Support", ["no", "yes"]),
    'internet': st.selectbox("Internet", ["yes", "no"]),
    'absences': st.number_input("Absences", 0, 93, 2),
    'G1': st.slider("G1 Score", 0, 20, 12),
    'G2': st.slider("G2 Score", 0, 20, 12),
}

# Your entire pipeline runs in 1 line:
if st.button("Predict"):
    pred = np.clip(model.predict(pd.DataFrame([data]))[0], 0, 20)
    st.success(f"Predicted G3: {pred:.2f} / 20")
