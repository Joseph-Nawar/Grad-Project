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

FACE_INTERIM_MANIFEST_PATH = INTERIM_DATA_DIR / "face_image_manifest.csv"
FACE_DUPLICATE_REPORT_PATH = INTERIM_DATA_DIR / "face_duplicate_report.csv"
FACE_CLEAN_MANIFEST_PATH = PROCESSED_DATA_DIR / "face_clean_manifest.csv"

FACE_SPLIT_MANIFEST_PATH = PROCESSED_DATA_DIR / "face_split_manifest.csv"
SPEECH_INTERIM_MANIFEST_PATH = INTERIM_DATA_DIR / "speech_audio_manifest.csv"
SPEECH_SPLIT_MANIFEST_PATH = PROCESSED_DATA_DIR / "speech_split_manifest.csv"
METADATA_SPLIT_MANIFEST_PATH = PROCESSED_DATA_DIR / "metadata_split_manifest.csv"

FACE_IMAGE_SIZE = (160, 160)
FACE_BATCH_SIZE = 16

FACE_EXPERIMENTS_DIR = MODELS_DIR / "experiments" / "face"

FACE_AUTOKERAS_TRIAL_001_DIR = FACE_EXPERIMENTS_DIR / "trial_001_autokeras_vanilla_160"
FACE_AUTOKERAS_TRIAL_001_MODEL_PATH = FACE_AUTOKERAS_TRIAL_001_DIR / "model_unusable.keras"

FACE_AUTOKERAS_TRIAL_001_RESULTS_PATH = REPORTS_DIR / "experiments" / "face_trial_001_autokeras_vanilla_160_results.md"
FACE_AUTOKERAS_TRIAL_001_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "experiments" / "face_trial_001_autokeras_vanilla_160_predictions.csv"
FACE_AUTOKERAS_TRIAL_001_CONFUSION_MATRIX_PATH = FIGURES_DIR / "experiments" / "face_trial_001_autokeras_vanilla_160_confusion_matrix.png"

FACE_AUTOKERAS_TRIAL_002_DIR = (
    MODELS_DIR / "experiments" / "face" / "trial_002_autokeras_broad_160"
)

FACE_AUTOKERAS_TRIAL_002_MODEL_PATH = (
    FACE_AUTOKERAS_TRIAL_002_DIR / "model.keras"
)

FACE_AUTOKERAS_TRIAL_002_RESULTS_PATH = (
    REPORTS_DIR / "experiments" / "face_trial_002_autokeras_broad_160_results.md"
)

FACE_AUTOKERAS_TRIAL_002_PREDICTIONS_PATH = (
    PROCESSED_DATA_DIR / "experiments" / "face_trial_002_autokeras_broad_160_predictions.csv"
)

FACE_AUTOKERAS_TRIAL_002_CONFUSION_MATRIX_PATH = (
    FIGURES_DIR / "experiments" / "face_trial_002_autokeras_broad_160_confusion_matrix.png"
)

FACE_AUTOKERAS_TRIAL_002_HISTORY_PATH = (
    PROCESSED_DATA_DIR / "experiments" / "face_trial_002_autokeras_broad_160_history.csv"
)

