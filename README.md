# Structured Approach for Building Scikit-Learn Pipelines

This repository demonstrates how to create machine learning pipelines in a slightly more organized, maintainable, and flexible way using Scikit-Learn. 
The project applies this structured approach to the House Pricing Kaggle competition dataset.
## Project Overview
The main goal of this repository is to illustrate how to build and structure Scikit-Learn pipelines effectively. 
By following a modular and systematic approach, the project aims to improve code readability, reusability, and maintainability while simplifying the machine learning workflow.

## Repository Structure
```
structured-approach-for-building-sklearn-pipelines/
│-- config/                    # Main modules
  |-- data.py                  # Module for loading data
  |-- nested_5x2_cv.py         # Implementation of nested cross validation for rough algorithm estimation
  |-- pipelines.py             # Module for simple and structured building of model pipelines
│-- data/                      # Data files
│-- figures/                   # Figures
│-- optunalogs/                # Optuna study logs
│-- submissions/               # Submissions for the competition
│-- optuna_optimization.py     # Module for hyperparameter search
│-- models_eval.py             # Module for model evaluation
|-- averaged_submission.py     # Module for creating averaged prediction from multiple submissions
```

## Notes
The purpose of this project is not to demonstrate how to build highly accurate models, although the best submission, which scored **12359.71**, helped me get into the top 50 (7%) of the leaderboard. 
If you want to improve this result, I can recommend you to focus on improving the XGB and LGBM models, as their individual scores is low.

