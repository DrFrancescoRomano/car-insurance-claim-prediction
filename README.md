# Car Insurance Claim Prediction

This project aims to build a predictive model to determine whether a customer will file an insurance claim during the policy period. The model uses logistic regression and feature optimization techniques to identify key factors influencing claims.

## Project Overview

Insurance companies are increasingly leveraging machine learning to optimize pricing and better assess the likelihood of claims. In this project, I developed a logistic regression model using customer data to predict the likelihood of an insurance claim. This can help insurance companies make informed decisions, adjust premiums, and improve risk management strategies.

## Data Description

The dataset consists of several features related to the customer’s demographics and driving history, including:
- `age`: Customer age in years
- `gender`: Gender of the customer
- `driving_experience`: Years of driving experience
- `vehicle_ownership`: Ownership status of the vehicle (1 = owned, 0 = not owned)
- `credit_score`: Customer's credit score
- `annual_mileage`: Average annual mileage
- `past_accidents`: Number of past accidents
- `speeding_violations`: Number of speeding violations
- `duis`: Number of DUI violations
- `outcome`: Target variable (1 = claim, 0 = no claim)

## Model Description

- **Model Type**: Logistic Regression
- **Evaluation Metrics**: Accuracy, ROC-AUC, Precision, Recall, F1 Score
- **Feature Optimization**: Feature selection through backward elimination

### Key Results:
- Best performing feature: `driving_experience`
- Model accuracy: **85.45%**
- Area under ROC curve (AUC): **0.909**


## Interactive Dashboard

In addition to the predictive model, I developed an interactive dashboard to provide insightful visualizations of the data. The dashboard is built using the `flexdashboard` package in R and displays key statistics and data visualizations to help insurance companies better understand the relationship between various features and insurance claims.

### Features of the Dashboard:
- **Visualizations**: Includes bar charts, scatter plots, and value boxes to display important metrics like mean income, credit score, and vehicle ownership.
- **Data Table**: Displays a comprehensive view of the data, allowing users to filter and explore different variables.
- **Interactive Tabs**: Users can navigate between different sections to explore data visualizations, summaries, and specific metrics related to the dataset.

### How to Access the Dashboard:
To view the interactive dashboard, open the `dashboard_carInsurance.html` file located in the repository. You can open the HTML file directly in any modern web browser to interact with the dashboard.

## Repository Structure

- `ModellingCarInsurance_.R`: Main R script used for data preprocessing, model training, and evaluation.
- `dashboard_carInsurance.html`: Interactive dashboard for visualizing key metrics and data insights.
- `README.md`: Project documentation.
- `.gitignore`: Specifies files to ignore in the repository.
- `car-insurance-claim-prediction.Rproj`: RStudio project file.




## Repository Structure

- `ModellingCarInsurance_.R`: Main R script used for data preprocessing, model training, and evaluation.
- `README.md`: Project documentation.
- `.gitignore`: Specifies files to ignore in the repository.
- `car-insurance-claim-prediction.Rproj`: RStudio project file.

## Installation and Usage

To run this project locally, you need to install the following R packages:
- `dplyr`
- `caret`
- `readr`
- `yardstick`
- `pROC`
- `boot`

To install these packages, run:

```r
install.packages(c("dplyr", "caret", "readr", "yardstick", "pROC", "boot"))
```

## Future Improvements

1. **Testing additional machine learning models**: Such as Random Forest or Gradient Boosting to compare performance.
2. **Feature Engineering**: Explore new features like interaction terms between demographics and driving history.
3. **Model Deployment**: Implement the model in a web application for real-time predictions.

## Contact

If you have any questions or suggestions, feel free to contact me:

- **Name**: Francesco Romano
- **LinkedIn**: [Francesco Romano](https://www.linkedin.com/in/francescoromano03/)
