## Credit Card Fraud Detection
This project is focused on detecting fraudulent credit card transactions using machine learning techniques. 
The model is trained on a dataset of credit card transactions and uses classification algorithms to
predict whether a transaction is fraudulent or legitimate.

## Features
- Fraud Detection: The model classifies credit card transactions as fraudulent or legitimate.
- Machine Learning Algorithms: Implemented using various classification algorithms such as Logistic Regression, Random Forest, Support Vector Machine, etc.
- Data Preprocessing: Includes data cleaning, feature engineering, and normalization.
- Model Evaluation: The model is evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## Requirements


- Python 3.x
## Libraries:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn (for handling imbalanced datasets)

1. Dataset
The dataset used for this project is Credit Card Fraud Detection Dataset from Kaggle.
 You can download the dataset and place it in the project folder as creditcard.csv.

Alternatively, if you already have the dataset in CSV format, just replace the filename in the code.

2. Explore the Dataset
You can start by exploring the dataset with the script data_exploration.py. This script will help you:

Load and clean the data
Handle missing values (if any)
Analyze and visualize transaction features

3. Model Training
Run the train_model.py script to train the fraud detection model. The script will:

Preprocess the data (including normalization and encoding)
Train various models (Logistic Regression, Random Forest, etc.)
Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score
