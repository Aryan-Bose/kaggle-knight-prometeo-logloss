import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# ---------------- LOAD DATA ----------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ---------------- CONFIG ----------------
TARGET_COL = "isFraud"
ID_COL = "id"
RANDOM_STATE = 42

# ---------------- FREQUENCY ENCODING ----------------
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns:", cat_cols)

for col in cat_cols:
    freq = train[col].value_counts()
    train[col] = train[col].map(freq)
    test[col] = test[col].map(freq).fillna(0)

# ---------------- PREPARE DATA ----------------
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# ---------------- XGBOOST MODEL (LOG LOSS SAFE) ----------------
model = XGBClassifier(
    n_estimators=5000,        # conservative
    learning_rate=0.01,
    max_depth=1,
    min_child_weight=1020,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=2.0,
    reg_lambda=10.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    tree_method="hist"
)

# ---------------- TRAIN ----------------
model.fit(X_train, y_train)

# ---------------- VALIDATION ----------------
val_preds = model.predict_proba(X_val)[:, 1]
val_ll = log_loss(y_val, val_preds)
print("Validation Log Loss:", val_ll)

# ---------------- TRAIN FULL ----------------
model.fit(X, y)

# ---------------- TEST PREDICTION ----------------
test_preds = model.predict_proba(test)[:, 1]

# ---------------- SUBMISSION ----------------
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    "prediction": test_preds
})

submission.to_csv("submission_xgboost_logloss.csv", index=False)
print("submission_xgboost_logloss.csv created successfully!")
