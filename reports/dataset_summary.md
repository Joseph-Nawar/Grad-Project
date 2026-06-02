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