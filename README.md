## Insurance Charges Prediction System (Version 2)

A professional End-to-End Machine Learning project that predicts individual medical insurance charges using Multiple Linear Regression with Log Transformation. The project includes complete data preprocessing, feature engineering, statistical analysis, model evaluation, and a user-friendly Tkinter GUI for real-time predictions.

## Project Overview

Healthcare insurance costs depend on several factors such as age, BMI, smoking habits, number of children, and geographical region. The objective of this project is to develop an accurate machine learning model capable of estimating insurance charges based on these attributes.

Version 2: enhances the original project by introducing Log Transformation on the target variable to reduce skewness and improve prediction accuracy. It also provides improved visualizations, better model evaluation, and a professional desktop application.

## Version 2 Improvements

-  Upgraded to Version 2
-  Implemented Log Transformation on the target variable (`charges`)
-  Reduced target variable skewness
-  Improved Multiple Linear Regression performance
-  Enhanced model evaluation
-  Improved data visualization
-  Professional Tkinter GUI
-  Better project structure and code organization

## Features

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Encoding
- Feature Engineering
- Feature Scaling using StandardScaler
- Pearson Correlation Analysis
- Chi-Square Statistical Analysis
- Multiple Linear Regression
- Log Transformation (`log1p`)
- Model Evaluation
- Synthetic Inference
- Professional GUI Application
- Insurance Charges Prediction

## Dataset

Dataset: `insuranceFH.csv` [Source: Kaggle]

The dataset contains demographic and health-related information used to estimate insurance charges.

## Technologies Used

## Programming Language

- Python 3.x

## Libraries

- NumPy
- Pandas
- Seaborn
- Matplotlib
- Scikit-Learn
- SciPy
- Pickle
- Tkinter
- Pillow

## Machine Learning Workflow

Data Collection
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Cleaning
        │
        ▼
Feature Encoding
        │
        ▼
Feature Engineering
        │
        ▼
Feature Scaling
        │
        ▼
Feature Selection
        │
        ▼
Log Transformation
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
GUI Prediction System

## Data Preprocessing

- Removed duplicate records
- Label Encoding
- One-Hot Encoding
- BMI Category creation
- StandardScaler for numerical features


## Feature Selection

## Pearson Correlation

Used to determine relationships between numerical features and insurance charges.

## Chi-Square Test

Used to evaluate the statistical significance of categorical variables.

## Why Log Transformation?

The target variable (`charges`) is highly right-skewed.

Applying Log Transformation:

- Reduces skewness
- Minimizes the influence of extreme values
- Improves Linear Regression assumptions
- Produces better predictions
- Increases model performance

## Machine Learning Model

- Multiple Linear Regression

## Model Performance

| Metric               |  Version 1 | Version 2 |
|----------------------|-----------:|----------:|
| R² Score (Test Data) |     0.8055 |    0.8278 |

## Improvement

- Better prediction accuracy
- Better model generalization
- Improved residual distribution
- More reliable predictions

## GUI Application

The project includes a professional desktop application developed using Tkinter.

## GUI Features

- User-friendly interface
- BMI Category calculation
- Real-time prediction
- Reset functionality
- Input validation
- Professional layout
- Status bar
- Company branding

## ▶ How to Run

## Clone Repository

```bash
git clone <repository-url>
```

### Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy pillow
```

### Train the Model

```bash
python LRLogTranfFH.py
```

### Run the GUI

```bash
python GUI.py
```

##  Future Enhancements

- Random Forest Regression
- XGBoost Regression
- Hyperparameter Tuning
- Cross Validation
- Feature Importance Analysis
- Streamlit Web Deployment
- Docker Containerization


## Author

Farhana Hameed
© 2026 Innovative Tech. All Rights Reserved.