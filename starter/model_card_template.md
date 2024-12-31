# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Xiaoxiao Qi created the model on 12/25/2024. 
This app uses a RandomForest model and predicts whether income exceeds $50k/yr based on personal census data.

## Intended Use
This model predicts the salary based on some personal features. You can run the predict through an deployed API.

## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The data is split in 80% of data for training phase and 20% for test phase.

## Evaluation Data

## Metrics
The model was evaluated using Precision, Recall and fbeta.
Current model scores are: `precision=0.670`, `Recall=0.571`, `FBeta=0.617`

## Ethical Considerations
Data related to race and gender.

## Caveats and Recommendations
Improve the model performance by parameter tunning and try different models.