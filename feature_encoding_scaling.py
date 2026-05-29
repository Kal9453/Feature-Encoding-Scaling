# ============================================================
# Task 4: Feature Encoding & Scaling
# AI & ML Internship - EduTech Solutions
# Dataset: Adult Census Income Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
)
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("TASK 4: FEATURE ENCODING & SCALING")
print("=" * 60)

# Adult Census Income Dataset column names
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week',
    'native_country', 'income'
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

try:
    df = pd.read_csv(url, names=columns, sep=',', skipinitialspace=True, na_values='?')
    print("\n✅ Dataset loaded from UCI repository.")
except Exception:
    # Fallback: generate a representative synthetic sample
    print("\n⚠️  Could not fetch from UCI. Generating representative synthetic data...")
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'age':           np.random.randint(18, 75, n),
        'workclass':     np.random.choice(['Private','Self-emp','Gov','Without-pay'], n,
                                           p=[0.70, 0.12, 0.15, 0.03]),
        'fnlwgt':        np.random.randint(12285, 1484705, n),
        'education':     np.random.choice(['Bachelors','Some-college','HS-grad',
                                            'Masters','Assoc','Doctorate'], n,
                                           p=[0.20,0.25,0.32,0.10,0.08,0.05]),
        'education_num': np.random.randint(1, 16, n),
        'marital_status':np.random.choice(['Married','Never-married',
                                            'Divorced','Separated','Widowed'], n,
                                           p=[0.46,0.32,0.14,0.04,0.04]),
        'occupation':    np.random.choice(['Tech-support','Craft-repair',
                                            'Sales','Exec-managerial',
                                            'Prof-specialty','Handlers-cleaners'], n),
        'relationship':  np.random.choice(['Wife','Own-child','Husband',
                                            'Not-in-family','Other-relative',
                                            'Unmarried'], n),
        'race':          np.random.choice(['White','Black','Asian-Pac-Islander',
                                            'Amer-Indian-Eskimo','Other'], n,
                                           p=[0.85,0.09,0.03,0.02,0.01]),
        'sex':           np.random.choice(['Male','Female'], n, p=[0.67,0.33]),
        'capital_gain':  np.where(np.random.random(n) > 0.9,
                                   np.random.randint(1000, 99999, n), 0),
        'capital_loss':  np.where(np.random.random(n) > 0.95,
                                   np.random.randint(100, 4356, n), 0),
        'hours_per_week':np.random.randint(1, 99, n),
        'native_country':'United-States',
        'income':        np.random.choice(['<=50K','>50K'], n, p=[0.75,0.25]),
    })

print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Statistics:\n{df.describe()}")

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: DATA CLEANING")
print("=" * 60)

df_clean = df.dropna()
print(f"Rows before cleaning : {len(df)}")
print(f"Rows after  cleaning : {len(df_clean)}")
df = df_clean.reset_index(drop=True)

# ─────────────────────────────────────────────
# 3. IDENTIFY VARIABLE TYPES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: IDENTIFY NOMINAL vs ORDINAL VARIABLES")
print("=" * 60)

# Ordinal: education has a natural order; encode with LabelEncoder
ordinal_features = ['education']

# Nominal: no intrinsic order — One-Hot Encode
nominal_features = ['workclass', 'marital_status', 'occupation',
                    'relationship', 'race', 'sex', 'native_country']

# Target
target_col = 'income'

# Numerical features (to be scaled)
numerical_features = ['age', 'fnlwgt', 'education_num',
                      'capital_gain', 'capital_loss', 'hours_per_week']

print(f"\nOrdinal  features : {ordinal_features}")
print(f"Nominal  features : {nominal_features}")
print(f"Numerical features: {numerical_features}")

# ─────────────────────────────────────────────
# 4. LABEL ENCODING  (ordinal)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: LABEL ENCODING (Ordinal Feature — education)")
print("=" * 60)

education_order = [
    'Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th',
    'HS-grad','Some-college','Assoc-voc','Assoc-acdm',
    'Bachelors','Masters','Prof-school','Doctorate'
]

le = LabelEncoder()
le.classes_ = np.array(education_order)

# Map values not in the defined order to nearest match
known = set(education_order)
df['education_encoded'] = df['education'].apply(
    lambda x: education_order.index(x) if x in known else -1
)

print("\nEducation Encoding Map:")
for i, edu in enumerate(education_order):
    count = (df['education'] == edu).sum()
    if count > 0:
        print(f"  {edu:20s} → {i}  (n={count})")

print(f"\nSample — original vs encoded:\n"
      f"{df[['education','education_encoded']].drop_duplicates().sort_values('education_encoded')}")

# Also encode target
df['income_encoded'] = (df['income'].str.strip() == '>50K').astype(int)
print(f"\nTarget encoding: <=50K → 0, >50K → 1")
print(df['income_encoded'].value_counts())

# ─────────────────────────────────────────────
# 5. ONE-HOT ENCODING  (nominal)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: ONE-HOT ENCODING (Nominal Features)")
print("=" * 60)

print("\n⚠️  DUMMY VARIABLE TRAP: Using drop_first=True to avoid multicollinearity.")
df_encoded = pd.get_dummies(
    df,
    columns=nominal_features,
    drop_first=True,   # ← avoids dummy-variable trap
    dtype=int
)

new_cols = [c for c in df_encoded.columns if c not in df.columns]
print(f"\nNew one-hot columns created: {len(new_cols)}")
print(f"Columns added (sample): {new_cols[:8]} ...")
print(f"\nDataFrame shape after OHE: {df_encoded.shape}")

# ─────────────────────────────────────────────
# 6. FEATURE SCALING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: FEATURE SCALING")
print("=" * 60)

X_num = df[numerical_features].copy()

# --- StandardScaler (Z-score)
std_scaler  = StandardScaler()
X_std       = pd.DataFrame(std_scaler.fit_transform(X_num),
                            columns=[f"{c}_std" for c in numerical_features])

# --- MinMaxScaler (Normalization)
mm_scaler   = MinMaxScaler()
X_mm        = pd.DataFrame(mm_scaler.fit_transform(X_num),
                            columns=[f"{c}_mm" for c in numerical_features])

print("\nBefore Scaling — Statistics:")
print(X_num.describe().round(2))

print("\nAfter StandardScaler — Statistics:")
print(X_std.describe().round(2))

print("\nAfter MinMaxScaler — Statistics:")
print(X_mm.describe().round(2))

# ─────────────────────────────────────────────
# 7. PLOTS — Before / After scaling comparison
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: VISUALISING DISTRIBUTIONS BEFORE & AFTER SCALING")
print("=" * 60)

features_to_plot = ['age', 'fnlwgt', 'capital_gain', 'hours_per_week']

fig, axes = plt.subplots(len(features_to_plot), 3,
                          figsize=(15, 4 * len(features_to_plot)))
fig.suptitle("Feature Distributions: Before vs After Scaling\n"
             "(Task 4 — EduTech AI & ML Internship)",
             fontsize=14, fontweight='bold', y=1.01)

for i, feat in enumerate(features_to_plot):
    # Original
    axes[i, 0].hist(X_num[feat], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[i, 0].set_title(f"{feat}\nOriginal", fontsize=10)
    axes[i, 0].set_xlabel("Value")
    axes[i, 0].set_ylabel("Frequency")

    # StandardScaler
    axes[i, 1].hist(X_std[f"{feat}_std"], bins=40, color='darkorange', edgecolor='white', alpha=0.85)
    axes[i, 1].set_title(f"{feat}\nStandardScaler (μ=0, σ=1)", fontsize=10)
    axes[i, 1].set_xlabel("Standardized Value")

    # MinMaxScaler
    axes[i, 2].hist(X_mm[f"{feat}_mm"], bins=40, color='seagreen', edgecolor='white', alpha=0.85)
    axes[i, 2].set_title(f"{feat}\nMinMaxScaler [0, 1]", fontsize=10)
    axes[i, 2].set_xlabel("Normalized Value")

plt.tight_layout()
plt.savefig("scaling_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: scaling_comparison.png")

# ─────────────────────────────────────────────
# 8. ENCODING SUMMARY PLOT
# ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle("Encoding Overview — Task 4", fontsize=13, fontweight='bold')

# Education label encoding distribution
edu_counts = df[['education','education_encoded']].drop_duplicates().sort_values('education_encoded')
axes2[0].barh(edu_counts['education'], edu_counts['education_encoded'],
              color='mediumpurple', edgecolor='white')
axes2[0].set_title("Label Encoding — Education (Ordinal)")
axes2[0].set_xlabel("Encoded Value")

# Income distribution (target)
income_counts = df['income_encoded'].value_counts()
axes2[1].bar(['<=50K (0)', '>50K (1)'], income_counts.values,
              color=['tomato', 'cornflowerblue'], edgecolor='white')
axes2[1].set_title("Target Encoding — Income")
axes2[1].set_ylabel("Count")
for bar, val in zip(axes2[1].patches, income_counts.values):
    axes2[1].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 5, str(val), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("encoding_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: encoding_overview.png")

# ─────────────────────────────────────────────
# 9. FINAL PREPARED DATASET
# ─────────────────────────────────────────────
# Build final ML-ready dataframe
final_df = df_encoded.copy()

# Replace raw numericals with standardized versions
for feat in numerical_features:
    final_df[feat] = X_std[f"{feat}_std"].values

# Drop columns no longer needed
drop_cols = ['education', 'income', 'education_encoded',
             'income_encoded', 'fnlwgt']  # fnlwgt is a census weight, often dropped
final_df.drop(columns=[c for c in drop_cols if c in final_df.columns],
              inplace=True)

# Add back the encoded target
final_df['income'] = df['income_encoded'].values

print(f"\nFinal ML-ready DataFrame shape: {final_df.shape}")
print(f"Columns (sample):\n{list(final_df.columns[:10])} ...")
final_df.to_csv("adult_census_processed.csv", index=False)
print("✅ Saved: adult_census_processed.csv")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Techniques Used & Rationale
─────────────────────────────
1. Label Encoding  → 'education' is ORDINAL (Preschool < HS-grad < Bachelors
   < Masters < Doctorate). Numeric order carries real meaning.

2. One-Hot Encoding → 'workclass', 'marital_status', 'occupation', etc. are
   NOMINAL. No natural order exists, so integer codes would imply false
   relationships. OHE creates independent binary columns.

3. Dummy Variable Trap → Avoided using drop_first=True in pd.get_dummies().
   With k categories, only (k-1) columns are needed; the dropped column is
   implied when all others equal 0.

4. StandardScaler (Z-score) → Preferred for algorithms assuming Gaussian
   distributions (Logistic Regression, SVM, PCA). Centers at 0, unit variance.
   Robust to outliers compared to MinMax.

5. MinMaxScaler (Normalization) → Scales all values to [0, 1]. Preferred for
   neural networks and KNN where Euclidean distance matters, and the feature
   distribution is NOT Gaussian. Sensitive to outliers.
""")
print("=" * 60)
print("✅ Task 4 Complete!")
