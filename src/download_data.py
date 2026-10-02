from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

TARGET = DATA_DIR / "HR_Employee_Attrition.csv"

URLS = [
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv",
]

if TARGET.exists():
    print(f"Dataset already exists: {TARGET}")
    raise SystemExit(0)

last_error = None
for url in URLS:
    try:
        print(f"Downloading: {url}")
        urlretrieve(url, TARGET)
        print(f"Saved to: {TARGET}")
        break
    except Exception as exc:
        last_error = exc
else:
    raise RuntimeError(
        "Dataset download failed. Download the IBM HR Analytics Employee Attrition "
        "CSV from the public source listed in README.md and save it as "
        f"{TARGET.name} in {DATA_DIR}."
    ) from last_error
