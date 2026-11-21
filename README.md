#
# 🎓 Student Final Grade (G3) Predictor

An end-to-end Machine Learning project that predicts a student's final exam score (`G3` on a 0–20 scale) based on academic history, study habits, and demographic factors, deployed as an interactive **Streamlit** web application.

## 📊 Project Overview & Results
Using the **UCI Student Performance Dataset** (`student-por.csv`, 649 records), I tested multiple preprocessing strategies and evaluated how prior academic performance (`G1`, `G2`) impacts predictive accuracy:
- **Baseline (Behavioral & Demographic features only):** `17.20%` Test $R^2$ | `±10.09%` MAE
- **Full Feature Set (Including `G1` & `G2` + Yeo-Johnson Pipeline):** **`85.98%` Test $R^2$** | **`±0.77` marks MAE** (on a 0–20 scale)

## 🧠 Key Concepts & Skills Learned
- **Exploratory Data Analysis (EDA):** Automated profiling (`ydata-profiling`), skewness/kurtosis analysis, and outlier detection using the IQR method.
- **Custom Scikit-Learn Transformers:** Implemented a custom `IQRCapper(BaseEstimator, TransformerMixin)` class to clip `age` and `absences` outliers dynamically inside cross-validation folds to prevent data leakage.
- **Preprocessing Pipelines:** Combined `SimpleImputer`, `IQRCapper`, `PowerTransformer` (Yeo-Johnson), `StandardScaler`, and `OneHotEncoder` using `ColumnTransformer` and `Pipeline`.
- **Model Validation:** Evaluated models using 10-fold `KFold` cross-validation and hold-out test splits across `MAE`, `RMSE`, and $R^2$.
- **Model Serialization & Deployment:** Exported the end-to-end fitted pipeline using `joblib` and built an interactive UI with **Streamlit**.

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib
- **Deployment:** Streamlit

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
