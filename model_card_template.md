# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Overview 
This is a logistic regression classifier that predicts individual income being more or less than $50k based on demographics. Such as age, education, occupation and hours worked per week.

## Inteded Use
This model can be used by organizations for predicting the income bracket of individual demographics. It could be used for studies with income disparities across different population groups.

## Dataset Overview
The model was trained using the Census Income Dataset. The dataset contains demographic information such as age, working class, ocupation, hours worked, and education.

## Evaluation Data
The model was evaluated using a test set from the same Census Income Dataset. It includes similar democraphic features as the training set.

## Metrics
-**Precision**: 0.7306
-**Recall**: 0.5646
-**F1 Score**: 0.6370
Metrics were selected to evaluate the model's performance identifying differences between individuals that earn more or less than $50K 

## Ethical Considerations
Potential biases from data inherited, such as ones based on **race** or **gender**. It may influence the model's predictions.

## Caveats and Recommendations
The predictions will be bias if the training data is also. Retrain model with more diverse data.