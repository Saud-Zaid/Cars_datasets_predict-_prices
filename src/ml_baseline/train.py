from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import pandas as pd

def build_model(df: pd.DataFrame, target: str) -> Pipeline:
    # تحديد الأعمدة الرقمية والفئوية تلقائياً
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns

    # تجهيز البيانات (Preprocessing)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    # بناء النموذج (Logistic Regression بدلاً من KMeans)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    return model
    


import json
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from .config import Paths
#from .model import build_model
from .matrics import get_classification_metrics

def train_model(target: str = "is_high_value", test_size: float = 0.2, seed: int = 42):
    paths = Paths.from_repo_root()
    
    input_path = paths.data_processed_dir / "features.csv"
    if not input_path.exists():
        input_path = paths.data_processed_dir / "features.parquet"
    
    print(f"Loading data from: {input_path}")
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset columns: {df.columns.tolist()}")


    X = df.drop(columns=[target, "user_id"], errors="ignore")
    y = df[target]

    print(f"Splitting data (seed={seed})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print("Training Baseline...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_base = dummy.predict(X_test)
    baseline_metrics = get_classification_metrics(y_test, y_pred_base)
    print(f"Baseline Metrics: {baseline_metrics}")

    # 5. Train Real Model [5]
    print("Training Model...")
    pipeline = build_model(df, target)
    pipeline.fit(X_train, y_train)
    
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{ts}_classification_seed{seed}"
    run_dir = paths.root / "models" / "runs" / run_id
    
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (paths.root / "models" / "registry").mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics" / "baseline_holdout.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)
        
    joblib.dump(pipeline, run_dir / "model" / "model.joblib")
    
    with open(paths.root / "models" / "registry" / "latest.txt", "w") as f:
        f.write(run_id)

    print(f"✅ Training complete! Run saved to: {run_dir}")
    print(f"✅ Baseline metrics saved to: metrics/baseline_holdout.json")
