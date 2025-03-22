# 🫀 Heart Disease Prediction with PyTorch

## 🚀 Project Overview
This repository contains a **heart disease prediction model** built using **PyTorch**, leveraging a **deep learning approach** for binary classification. The model predicts whether a person has heart disease based on various health-related features.

---

## 📂 Dataset
- The dataset is loaded from `heart.csv`.
- It includes features such as **age, sex, cholesterol levels, blood pressure, and more**.
- The target variable (`target`) indicates **1 for presence** of heart disease and **0 for absence**.

---

## ⚙️ Data Preprocessing
✅ **Handling Missing Values**: Replaces missing values with the column mean.<br>
✅ **Feature Scaling**: Uses `StandardScaler` to normalize the dataset.<br>
✅ **Data Splitting**: 80% training, 20% testing using `train_test_split`.<br>
✅ **Tensor Conversion**: Converts NumPy arrays into PyTorch tensors.

---

## 🏗 Model Architecture
This model follows a **4-layer neural network**:

| Layer  | Type      | Units |
|--------|----------|-------|
| Input  | Linear   | `n_features` |
| Hidden | Linear + ReLU | 64 |
| Hidden | Linear + ReLU | 32 |
| Hidden | Linear + ReLU | 16 |
| Output | Linear + Sigmoid | 1 |

- **ReLU (Rectified Linear Unit)** is used for non-linearity.
- **Sigmoid** activation at the output for binary classification.

---

## 🏋️ Training Process
🔹 **Loss Function**: Binary Cross-Entropy Loss (`BCELoss`)<br>
🔹 **Optimizer**: Adam Optimizer with `learning_rate = 0.0001`<br>
🔹 **Epochs**: Trained for `10,000` iterations.<br>
🔹 **Loss is printed every 1,000 epochs** for monitoring.

---

## 🎯 Evaluation
After training, the model is tested on the unseen test dataset:
- Predictions are rounded to **0 or 1**.
- Accuracy is computed using:
  ```python
  acc = y_pred_results.eq(y_test).sum() / float(y_test.shape[0])
  ```
- Final accuracy is printed after evaluation.

---

## 📌 How to Run
1️⃣ Clone this repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-predictor.git
   ```
2️⃣ Install dependencies:
   ```bash
   pip install pandas numpy torch scikit-learn
   ```
3️⃣ Run the script:
   ```bash
   python heart_disease_model.py
   ```

---

## 📊 Results
The model successfully predicts heart disease with **high accuracy**, making it a valuable tool for medical predictions.

📝 **Future Improvements:**
- Implement **hyperparameter tuning**.
- Introduce **dropout layers** to prevent overfitting.
- Try **different ML models** (e.g., Random Forest, SVM) for comparison.

---

## 👨‍💻 Author
**Surya Pasupuleti**
📌 **Machine Learning & AI Enthusiast**  
📩 Contact: [your-email@example.com](mailto:your-email@example.com)

---

## ⭐ Contributions & Support
💡 If you like this project, consider **starring** 🌟 the repository!  
🐛 Found an issue? Feel free to **open an issue** or submit a **pull request**.

---

### 🏆 "AI for a Healthier Future"

