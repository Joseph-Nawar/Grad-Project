from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FACE_DATA_DIR = RAW_DATA_DIR / "face_stroke_images"
METADATA_DATA_DIR = RAW_DATA_DIR / "stroke_prediction"

FACE_KAGGLE_DATASET = "abdussalamelhanashy/annotated-facial-images-for-stroke-classification"
METADATA_KAGGLE_DATASET = "fedesoriano/stroke-prediction-dataset"

# Manifest file paths
FACE_IMAGE_MANIFEST_PATH = INTERIM_DATA_DIR / "face_image_manifest.csv"

# Figures and plots paths
METADATA_DISTRIBUTION_PLOT_PATH = FIGURES_DIR / "metadata_class_distribution.png"

# Metadata EDA CSV paths
METADATA_CLASS_BALANCE_PATH = INTERIM_DATA_DIR / "metadata_class_balance.csv"
METADATA_MISSING_VALUES_PATH = INTERIM_DATA_DIR / "metadata_missing_values.csv"

SPEECH_DATA_DIR = RAW_DATA_DIR / "torgo_audio"

SPEECH_KAGGLE_DATASET = "pranaykoppula/torgo-audio"

# Test reports paths
SMOKE_TEST_REPORT_PATH = REPORTS_DIR / "milestone_1_smoke_test_report.txt"
