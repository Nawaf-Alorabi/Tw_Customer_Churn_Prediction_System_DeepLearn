# 📊 Customer Churn Prediction using Deep Learning

## 📌 Project Overview

This project focuses on predicting customer churn in a telecom company using deep learning techniques on the Telco Customer Churn dataset (7,043 records × 21 features).

The main goal is to identify customers who are likely to leave the service, so that businesses can take early actions to improve customer retention.

This work was carried out as a **team-based experimental project**, where each team member developed and tested their own notebook independently using different approaches. After that, the results were compared, and the best-performing model was selected based on evaluation metrics.

---

## 👥 Team Members & Individual Approaches

| Member | Approach | Key Technique |
|--------|----------|---------------|
| **Abdulrahman** | Supervised ANN with 5-Fold CV & threshold tuning | Class weights, BatchNorm, ReduceLROnPlateau |
| **Faisal** | Supervised ANN with 5-Fold CV & 3 optimizers | L2 + Dropout, EarlyStopping, 48 HP combos |
| **Nawaf** | Supervised ANN with 5-Fold CV & SMOTE | SMOTE oversampling, BatchNorm, threshold tuning |
| **Mohamed** | Unsupervised Autoencoder for anomaly detection | Denoising AE, Gaussian noise injection, L2 + Dropout |

### Team Strategy

Each member worked on a separate notebook with a different configuration or approach. The supervised ANN notebooks (Abdulrahman, Faisal, Nawaf) were directly compared using classification metrics. Mohamed explored an alternative unsupervised approach using autoencoders, treating potential churners as anomalies based on reconstruction error.

After comparing all results, the best-performing supervised model was selected based on F1 Score, Recall, and overall metric balance.

---

## 🎯 Objectives

- Build and compare multiple deep learning models for churn prediction
- Explore both supervised (ANN classifiers) and unsupervised (Autoencoder) approaches
- Test different architectures, hyperparameters, regularization techniques, and imbalance handling strategies
- Use cross-validation for robust hyperparameter selection
- Optimize the classification threshold for imbalanced data
- Select the best model based on comprehensive evaluation metrics

---

## ⚙️ Tools and Libraries

- Python
- TensorFlow / Keras
- Pandas & NumPy
- Scikit-learn (metrics, cross-validation, preprocessing)
- imbalanced-learn (SMOTE — Nawaf's notebook)
- Matplotlib & Seaborn

---

## 📊 Results Comparison — Supervised ANN Models

### Advanced Model Results (Final Test Set)

| Metric | Abdulrahman | Faisal | Nawaf |
|--------|:-----------:|:------:|:-----:|
| Accuracy | **0.7757** | 0.7637 | 0.7452 |
| Precision | **0.5585** | 0.5408 | 0.5134 |
| Recall | 0.7406 | 0.7273 | **0.7701** |
| F1 Score | **0.6368** | 0.6203 | 0.6160 |
| ROC-AUC | 0.8399 | **0.8446** | 0.8328 |

### Baseline Model Results

| Metric | Abdulrahman | Faisal | Nawaf |
|--------|:-----------:|:------:|:-----:|
| Accuracy | 0.7857 | **0.7913** | 0.7807 |
| Precision | **0.6233** | 0.6190 | 0.5886 |
| Recall | 0.4866 | **0.5562** | 0.5775 |
| F1 Score | 0.5465 | **0.5859** | 0.5830 |
| ROC-AUC | 0.8331 | **0.8441** | 0.8299 |

### Mohamed — Autoencoder Approach

Mohamed used a different methodology: an **unsupervised denoising autoencoder** that learns to reconstruct normal customer behavior, then flags customers with high reconstruction error as anomalies (potential churners). This approach does not produce the same classification metrics as the supervised models, but provides a complementary anomaly detection perspective. The autoencoder used L2 regularization, Dropout (0.2), Gaussian noise injection, and EarlyStopping.

---

## 🔍 Key Differences Between Approaches

### Imbalance Handling
- **Abdulrahman & Faisal:** Class weights (penalize missed churners ~2.8x more in the loss function — clean, doesn't alter data)
- **Nawaf:** SMOTE (generates synthetic minority samples to balance training set — inflates data, risk of learning synthetic patterns)
- **Mohamed:** Not applicable (unsupervised approach)

### Architecture & Regularization
- **Abdulrahman:** 128→64→32→1 with BatchNorm + L2 + Dropout (most regularization layers)
- **Faisal:** 128→64→32→1 with L2 + Dropout (no BatchNorm)
- **Nawaf:** 128→64→32→1 with BatchNorm + L2 + Dropout (same as Abdulrahman)
- **Mohamed:** Autoencoder: Input→32→16→32→Input with L2 + Dropout + Gaussian noise

### Hyperparameter Tuning
- **Abdulrahman:** 16 combos × 5-Fold CV = 80 runs, scoring by (F1+AUC)/2
- **Faisal:** 48 combos × 5-Fold CV = 240 runs (added 3 optimizers: Adam, RMSprop, SGD)
- **Nawaf:** 16 combos × 5-Fold CV = 80 runs (same structure, run on SMOTE-balanced data)
- **Mohamed:** Manual tuning with EarlyStopping (patience=3)

### Threshold Optimization
- **Abdulrahman:** Optimized via Precision-Recall curve → threshold = 0.4931
- **Faisal:** Default threshold = 0.5 (no optimization)
- **Nawaf:** Optimized via Precision-Recall curve on validation set → threshold = 0.3938
- **Mohamed:** Percentile-based reconstruction error threshold (95th percentile)

### Callbacks
- **Abdulrahman:** EarlyStopping (patience=7) + ReduceLROnPlateau (patience=3)
- **Faisal:** EarlyStopping (patience=5) only
- **Nawaf:** EarlyStopping (patience=4) + ReduceLROnPlateau (patience=3)
- **Mohamed:** EarlyStopping (patience=3)

---

## 🏆 Selected Model — Abdulrahman's Notebook

Abdulrahman's model was selected as the best-performing configuration based on the following reasons:

**Highest F1 Score (0.6368)** — the best balance between Precision and Recall, which is the most meaningful metric for churn prediction where both false positives and false negatives carry business cost.

**Strong Recall (0.7406)** — catches 74% of actual churners. While Nawaf's recall is slightly higher (0.7701), it comes at the cost of significantly lower Precision (0.5134 vs 0.5585) and the worst Accuracy (0.7452), indicating more false alarms.

**Best generalization** — the Val-Test accuracy gap is 4.8%, compared to Nawaf's 8.4% gap which suggests SMOTE-induced overfitting to synthetic training data.

**Clean imbalance handling** — class weights adjust the loss function without creating artificial data points, avoiding the generalization risks that SMOTE introduces.

**Most robust training strategy** — EarlyStopping with higher patience (7 epochs) + ReduceLROnPlateau allows the model more room to converge and fine-tune, compared to Faisal's shorter patience without LR scheduling.

**Production-ready inference pipeline** — includes a reusable `predict_churn()` function with the optimal threshold (0.4931) baked in, handling both single and batch predictions with human-readable output.

### Selected Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 128 → 64 → 32 → 1 |
| Regularization | L2 (0.01) + BatchNorm + Dropout (0.3) |
| Optimizer | Adam (LR = 0.001) |
| Imbalance | Class weights {0: 0.681, 1: 1.884} |
| Callbacks | EarlyStopping (patience=7) + ReduceLR (patience=3) |
| Threshold | 0.4931 (optimized via PR curve) |
| CV Score | 0.7342 (F1+AUC)/2 across 5 folds |

### Selected Model Final Results

| Metric | Baseline | Advanced |
|--------|:--------:|:--------:|
| Accuracy | 0.7857 | 0.7757 |
| Precision | 0.6233 | 0.5585 |
| Recall | 0.4866 | **0.7406** |
| F1 Score | 0.5465 | **0.6368** |
| ROC-AUC | 0.8331 | **0.8399** |

The advanced model traded a small amount of accuracy and precision for a **massive improvement in recall** (+25.4 percentage points), which is the priority in churn prediction — missing a churner is more costly than a false alarm.

---

## 📉 Visualizations Included (in Selected Notebook)

- Target distribution (count + pie chart)
- MonthlyCharges & tenure distributions by churn status
- Churn rate by contract type
- Training history curves (Loss, Accuracy, AUC)
- Precision-Recall vs Threshold analysis
- Confusion matrix heatmaps (Baseline vs Advanced)
- ROC curve comparison
- Grouped bar chart — all metrics side by side

---

## 🚀 Conclusion

By dividing the experiments across four notebooks — three supervised ANN classifiers and one unsupervised autoencoder — the team explored multiple deep learning approaches to churn prediction. Each member contributed unique techniques: different optimizers (Faisal), SMOTE balancing (Nawaf), anomaly detection via autoencoders (Mohamed), and comprehensive regularization with threshold tuning (Abdulrahman).

The selected model (Abdulrahman's) achieves the best F1 Score (0.6368) with strong Recall (0.7406), robust generalization, and a production-ready inference pipeline. It is the most suitable for real-world churn prediction because it reliably identifies likely churners while maintaining acceptable precision, supporting better business decisions and retention planning.

---

## 👤 Contributors

- **Abdulrahman** — Team Lead, ANN with cross-validation, BatchNorm, threshold optimization (selected model)
- **Faisal** — ANN with 3-optimizer search, L2 + Dropout, class weights
- **Nawaf** — ANN with SMOTE, BatchNorm, threshold optimization on validation set
- **Mohamed** — Autoencoder for anomaly detection, denoising approach with Gaussian noise
