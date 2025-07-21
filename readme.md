# 💰 Loan Default Prediction App

A machine learning project for predicting loan default risk using advanced classification models and a user-friendly Streamlit web interface.

---

## 📁 Project Structure

```
dl-loan-default-logistic-reg/
│
├── requirements.txt
├── readme.md
│
├── src/
│   ├── app.py
│   ├── cleaned_loan_df.pkl
│   ├── encoder.pkl
│   ├── scaler.pkl
│   ├── final_pipeline_with_meta.pkl
│   ├── features_config.pkl
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   ├── y_test.pkl
│   └── (other scripts or model files)
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── (other notebooks)
│
├── output/
│   ├── roc_curve.png
│   └── Target_value_count.png, other files
   

---

## 🚀 Features

- **Data Preprocessing:** Cleans and encodes loan data for modeling.
- **Model Training:** Uses Logistic Regression, Random Forest, and XGBoost with Optuna hyperparameter optimization.
- **Imbalanced Data Handling:** ADASYN oversampling for better minority class prediction.
- **Model Evaluation:** Generates metrics and ROC curve visualizations.
- **Web App:** Interactive Streamlit app for real-time loan default prediction.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/shakil-ai-lab/dl-loan-default-logistic-reg.git
   cd dl-loan-default-logistic-reg
   ```

2. **Create and activate a virtual environment (recommended):**
   ```
   python -m venv venv
   venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

---

## 🏃‍♂️ Usage

### 1. **Model Training & Evaluation**

- Use the Jupyter notebooks in the `notebooks/` folder to preprocess data, train models, and evaluate performance.
- Outputs such as trained models and metrics are saved in the `src/` and `output/` folders.

### 2. **Run the Streamlit App**

- From the `src/` folder, launch the app:
  ```
  streamlit run app.py
  ```
- The app will open in your browser. Enter applicant information to get a loan default prediction and probability.

---
## 📈 Model Performance

- **Recall:** 0.69  
- **Precision:** 0.22  
- **F1 Score:** 0.33  

These metrics were obtained on the test set using the best model pipeline.

## 📊 Notebooks

- **04_model_evaluation.ipynb:**  
  Contains code for loading models, evaluating performance, and visualizing results.

---

## 📦 Main Files

- `src/app.py` — Streamlit web application.
- `src/final_pipeline_with_meta.pkl` — Trained model pipeline with metadata.
- `src/encoder.pkl`, `src/scaler.pkl` — Preprocessing objects.
- `src/features_config.pkl` — Feature configuration for the app.
- `notebooks/` — Jupyter notebooks for data science workflow.
- `requirements.txt` — All required Python packages.

---

## ⚠️ Troubleshooting

- **Version Mismatch:**  
  Ensure you use the same scikit-learn version for saving and loading models.  
  Check your version with:
  ```python
  import sklearn
  print(sklearn.__version__)
  ```
- **Streamlit Warnings:**  
  Always run the app with `streamlit run app.py`, not `python app.py`.

---

## 📚 Dependencies

Key packages (see `requirements.txt` for full list):

- scikit-learn
- imbalanced-learn
- xgboost
- optuna
- pandas
- streamlit
- matplotlib
- joblib

---

## ✨ Credits

Developed by [Shakil Ur Rehman].  
For questions or contributions, please open an issue or pull request.

---

## 🙏 Request for Help & Collaboration

> ⚠️ **Note:** The current model has a relatively low precision score. If you have suggestions to improve the model’s performance—especially precision—I would greatly appreciate your insights or contributions.  
>
> Feel free to open an issue or submit a pull request.  
>  
> Thank you in advance for your support! 🙌

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
