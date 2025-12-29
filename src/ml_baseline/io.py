import pandas as pd
from pathlib import Path

def parquet_supported() -> bool:
    try:
        import pyarrow
        return True
    except ImportError:
        return False

def write_tabular(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        if not parquet_supported():
            raise ImportError("pyarrow not installed")
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
