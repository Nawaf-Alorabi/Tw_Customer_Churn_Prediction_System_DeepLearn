# 📊 Customer Churn Prediction using ANN

## 📌 Project Overview
This project focuses on predicting customer churn using Artificial Neural Networks (ANN).  
The main goal is to identify customers who are likely to leave the service, so that businesses can take early actions to improve customer retention.

This work was carried out as a **team-based experimental project**, where each team member developed and tested their own ANN notebook independently.  
After that, the results were compared, and the best-performing model was selected based on evaluation metrics.

---

## 👥 Team Work Strategy
The project was completed collaboratively by:

- Faisal
- Nawaf
- Abdulrahman
- Mohammed

### Team approach:
- Each member worked on a separate notebook
- Different ANN configurations and improvements were tested independently
- Results were compared across all notebooks
- The best model was selected based on performance

This approach helped us explore multiple ideas and choose the most effective ANN configuration.

---

## 🎯 Objectives
- Build ANN models for churn prediction
- Test different model architectures and hyperparameters
- Handle class imbalance
- Compare model results across team notebooks
- Select the best-performing model
- Evaluate the final model using multiple metrics

---

## 🧠 Final Selected Model
The selected model was the one that achieved the best balance in performance after comparing all team experiments.

It included improvements such as:
- Hyperparameter tuning
- Dropout
- L2 regularization
- EarlyStopping
- Class weights for imbalanced data

---

## ⚙️ Tools and Libraries
- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 📊 Evaluation Metrics
The models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 📈 Selected Model Results

| Model     | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------|---------|----------|--------|----------|---------|
| Baseline | 0.79    | 0.62     | 0.56   | 0.59     | 0.84    |
| Advanced | 0.76    | 0.54     | 0.73   | 0.62     | 0.84    |

### Key Insight
The advanced model achieved a clear improvement in **recall**, which is very important in churn prediction because identifying customers who may leave is more critical than reducing false positives.

Although precision decreased slightly, the overall balance improved, as shown by the higher **F1-score**.

---

## 📉 Visualizations Included
- Training vs Validation Loss
- Training vs Validation Accuracy
- Baseline vs Advanced comparison charts
- Radar chart

---

## 🚀 Conclusion
By dividing the experiments across multiple notebooks and comparing results as a team, we were able to explore different ANN approaches efficiently and select the best-performing churn prediction model.

This final model is more suitable for churn prediction because it improves the detection of likely churn customers, which supports better business decisions and retention planning.

---

## 👤 Contributors
- Faisal
- Nawaf
- Abdulrahman
- Mohammed
