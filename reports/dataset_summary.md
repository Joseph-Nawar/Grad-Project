# Dataset Summary

## Face Dataset

Dataset: Annotated Facial Images for Stroke Classification

Total images:
3749

Class balance:
stroke      count	percentage
NonStroke	2500	66.68
Stroke	    1249	33.32

Preprocessing issues:
- class imbalance
- image size variation
- face positioning variation
- lighting variation
- possible sourcing limitations

## Speech Dataset

Dataset: TORGO Audio Dataset

Total `.wav` files found:
17,635

Current readable audio files:
17,633

Class balance after corrected TORGO folder mapping:

- Control: 11,455 readable files
- Dysarthric: 6,178 readable files

Raw folder counts before readability filtering:

- F_Con: 4,677
- M_Con: 6,778
- F_Dys: 2,391
- M_Dys: 3,787

Preprocessing issues:
- TORGO is a dysarthria dataset, not a stroke-specific dataset.
- Dysarthric speech is used as a proxy for FAST-style speech abnormality.
- Speaker-level splitting is required to avoid leakage.
- Audio files vary in duration.
- Very short clips may be unusable.
- Long clips may require trimming or segmentation.
- Audio should be resampled to a consistent sample rate before modeling.

## Metadata Dataset

Dataset: Stroke Prediction Dataset

Total records:
5110

Class balance:
	
stroke      count	percentage		
0          	4861	95.13
1	        249	    4.87

Preprocessing issues:
- severe class imbalance
- missing BMI values
- categorical encoding required
- feature scaling required
- not an acute triage dataset

## Project Relevance

The face dataset supports the facial analysis module.  
The metadata dataset supports the contextual risk module.  
Together, they provide the first implementation foundation for the multimodal architecture.



### Face Dataset Duplicate Audit

A duplicate audit was performed using SHA256 file hashing.

Findings:

- Total image files before cleaning: 3749
- Unique image hashes: 2116
- Duplicate extra files: 1633
- Duplicate extra percentage: 43.56%
- Cross-class duplicate hashes: 1
- Clean images after deduplication: 2115
- Removed files: 1634

Cleaning strategy:

- Raw files were left unchanged in `data/raw`.
- Exact duplicate files were identified using file hashes.
- Cross-class duplicates were removed entirely due to conflicting labels.
- Same-class duplicates were reduced to one representative image per unique hash.
- The cleaned modeling manifest was saved to `data/processed/face_clean_manifest.csv`.

This cleaning step is necessary because duplicate images can cause data leakage and artificially inflate model performance.