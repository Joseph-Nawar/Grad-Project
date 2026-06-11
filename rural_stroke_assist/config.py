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



FACE_TRIAL_003_DIR = (
    MODELS_DIR / "experiments" / "face" / "trial_003_mobilenetv2_balanced_160"
)

FACE_TRIAL_003_MODEL_PATH = FACE_TRIAL_003_DIR / "model.keras"

FACE_TRIAL_003_RESULTS_PATH = (
    REPORTS_DIR / "experiments" / "face_trial_003_mobilenetv2_balanced_160_results.md"
)

FACE_TRIAL_003_PREDICTIONS_PATH = (
    PROCESSED_DATA_DIR / "experiments" / "face_trial_003_mobilenetv2_balanced_160_predictions.csv"
)

FACE_TRIAL_003_CONFUSION_MATRIX_PATH = (
    FIGURES_DIR / "experiments" / "face_trial_003_mobilenetv2_balanced_160_confusion_matrix.png"
)

FACE_TRIAL_003_HISTORY_PATH = (
    PROCESSED_DATA_DIR / "experiments" / "face_trial_003_mobilenetv2_balanced_160_history.csv"
)

FACE_ERROR_ANALYSIS_DIR = (
    REPORTS_DIR / "error_analysis"
)

FACE_FALSE_POSITIVES_PATH = (
    FACE_ERROR_ANALYSIS_DIR / "face_trial_003_false_positives.csv"
)

FACE_FALSE_NEGATIVES_PATH = (
    FACE_ERROR_ANALYSIS_DIR / "face_trial_003_false_negatives.csv"
)

FACE_LOW_CONFIDENCE_CORRECT_PATH = (
    FACE_ERROR_ANALYSIS_DIR / "face_trial_003_low_confidence_correct.csv"
)

FACE_HIGH_CONFIDENCE_ERRORS_PATH = (
    FACE_ERROR_ANALYSIS_DIR / "face_trial_003_high_confidence_errors.csv"
)

FACE_ERROR_ANALYSIS_REPORT_PATH = (
    FACE_ERROR_ANALYSIS_DIR / "face_trial_003_error_analysis.md"
)

# Multiclass facial expression dataset: FER2013
FER2013_DATA_DIR = RAW_DATA_DIR / "fer2013"
FER2013_KAGGLE_DATASET = "msambare/fer2013"

FER2013_INTERIM_MANIFEST_PATH = (
    INTERIM_DATA_DIR / "fer2013_image_manifest.csv"
)

FER2013_SPLIT_MANIFEST_PATH = (
    PROCESSED_DATA_DIR / "fer2013_split_manifest.csv"
)

FER2013_IMAGE_SIZE = (160, 160)
FER2013_BATCH_SIZE = 32
FER2013_RANDOM_STATE = 42

FER2013_TRIAL_001_DIR = (
    MODELS_DIR / "experiments" / "fer2013" / "trial_001_mobilenetv2_160"
)

FER2013_TRIAL_001_MODEL_PATH = FER2013_TRIAL_001_DIR / "model.keras"

FER2013_TRIAL_001_RESULTS_PATH = (
    REPORTS_DIR / "experiments" / "fer2013_trial_001_mobilenetv2_160_results.md"
)

FER2013_TRIAL_001_PREDICTIONS_PATH = (
    PROCESSED_DATA_DIR / "experiments" / "fer2013_trial_001_mobilenetv2_160_predictions.csv"
)

FER2013_TRIAL_001_CONFUSION_MATRIX_PATH = (
    FIGURES_DIR / "experiments" / "fer2013_trial_001_mobilenetv2_160_confusion_matrix.png"
)

FER2013_TRIAL_001_HISTORY_PATH = (
    PROCESSED_DATA_DIR / "experiments" / "fer2013_trial_001_mobilenetv2_160_history.csv"
)


SPEECH_SPLIT_MANIFEST_PATH = PROCESSED_DATA_DIR / "speech_split_manifest.csv"
SPEECH_FEATURES_PATH = PROCESSED_DATA_DIR / "speech_features.csv"

SPEECH_TRIAL_001_DIR = MODELS_DIR / "experiments" / "speech" / "trial_001_mfcc_random_forest"
SPEECH_TRIAL_001_MODEL_PATH = SPEECH_TRIAL_001_DIR / "model.pkl"

SPEECH_TRIAL_001_RESULTS_PATH = REPORTS_DIR / "experiments" / "speech_trial_001_mfcc_random_forest_results.md"
SPEECH_TRIAL_001_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "experiments" / "speech_trial_001_mfcc_random_forest_predictions.csv"
SPEECH_TRIAL_001_CONFUSION_MATRIX_PATH = FIGURES_DIR / "experiments" / "speech_trial_001_mfcc_random_forest_confusion_matrix.png"

SPEECH_FEATURE_EXTRACTION_FAILURES_PATH = (
    PROCESSED_DATA_DIR / "speech_feature_extraction_failures.csv"
)


SPEECH_TRIAL_002_DIR = (
    MODELS_DIR / "experiments" / "speech" / "trial_002_pycaret_benchmark"
)

SPEECH_TRIAL_002_MODEL_PATH = SPEECH_TRIAL_002_DIR / "model"

SPEECH_TRIAL_002_RESULTS_PATH = (
    REPORTS_DIR / "experiments" / "speech_trial_002_pycaret_benchmark_results.md"
)

SPEECH_TRIAL_002_LEADERBOARD_PATH = (
    PROCESSED_DATA_DIR / "experiments" / "speech_trial_002_pycaret_leaderboard.csv"
)

SPEECH_TRIAL_002_PREDICTIONS_PATH = (
    PROCESSED_DATA_DIR / "experiments" / "speech_trial_002_pycaret_predictions.csv"
)

SPEECH_TRIAL_002_CONFUSION_MATRIX_PATH = (
    FIGURES_DIR / "experiments" / "speech_trial_002_pycaret_confusion_matrix.png"
)