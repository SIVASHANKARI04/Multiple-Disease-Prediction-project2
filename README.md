
# Multiple-Disease-Prediction-project2

Here this link check it out:http://43.204.100.227:8502/

**About the Datasets**

**1. Parkinson’s Disease Dataset**

 Source:The dataset contains vocal measurements from individuals, including both healthy participants and those with Parkinson’s disease.
 
 **Features:**
  Vocal fundamental frequencies (MDVP:Fo, MDVP:Fhi, MDVP:Flo)
  Signal measures like jitter, shimmer, and HNR (Harmonic-to-Noise Ratio)
  Dynamical features such as RPDE, DFA, and pitch period entropy (PPE)
  Objective: To classify whether an individual is positive or negative for Parkinson’s disease.
  
 **2. Liver Disease Dataset**
Source: This dataset includes clinical and laboratory data of individuals, focusing on liver health.
**Features:**

Total Bilirubin, Direct Bilirubin
Alkaline Phosphatase, Alanine Aminotransferase, Aspartate Aminotransferase
Total Proteins, Albumin, and Albumin-Globulin ratio
Objective: To predict the presence or absence of liver disease.

**3. Kidney Disease Dataset**

 A medical dataset containing data on kidney function indicators.
**Features:**

Age, blood pressure (BP), specific gravity (SG), albumin (AL), sugar levels
Counts for red and white blood cells, serum creatinine, and sodium levels
Appetite status and coronary artery disease presence
Objective: To identify individuals at risk for chronic kidney disease.

**About the Models**

**Parkinson’s Disease Model**

**Algorithm:** XGBoost Classifier
**Preprocessing:** Feature scaling using a StandardScaler.
**Training:** Model trained on balanced data to optimize prediction accuracy.
**Evaluation:** Achieved high accuracy and F1-score, making it suitable for reliable predictions.

**Liver Disease Model**
**Algorithm:** Logistic Regression (or another appropriate model depending on the implementation).
**Preprocessing:** Scaled the features using a StandardScaler for consistent input representation.
**Training:** Tuned for sensitivity and specificity to handle imbalanced datasets.

**Kidney Disease Model**

**Algorithm:** Random Forest Classifier
**Preprocessing:** Similar scaling applied for numerical stability.
**Training:** Model fine-tuned for early-stage kidney disease detection with high precision.

**Why These Models?**

**XGBoost:** Ideal for structured datasets with its ability to handle missing data and boost prediction accuracy.

**Logistic Regression:** Effective for binary classification tasks with interpretable coefficients.

**Random Forest:** Robust against overfitting and excellent for handling non-linear relationships.
