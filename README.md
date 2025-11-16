# job-admission-logistic-regression
# Project: Job Admission Prediction with Logistic Regression

## I. Project Description

This project implements a Binary Classification model using the Logistic Regression method, developed as part of the Machine Learning course.  
The goal is to determine the job acceptance (`1`) or rejection (`0`) probability of applicants based on two exam scores.

**Key Requirement:**  
The core components of the Logistic Regression algorithm (Sigmoid function, Cross-Entropy Loss, and Stochastic Gradient Descent for weight update) are implemented **from scratch** using only **NumPy**, without relying on high-level built-in functions from libraries like Scikit-learn or TensorFlow.

---

## II. Project Structure

The project follows a modular structure as required by the assignment, separating the training logic from the evaluation logic.

````text
PROJECT_FOLDER/
├── dataset/
│   └── hw1Data.txt          # exam scores and target labels
├── results/                 # outputs: metrics, trained parameters, graphs
│   ├── final_theta.npy      # trained model weights
│   ├── loss_history.png     # loss graph
│   └── metrics_results.txt  # accuracy, precision, recall, f1-score
├── train.py                 # model training and parameter saving
├── eval.py                  # performance evaluation and metric calculation
└── requirements.txt
````


III. Execution Requirements

1. Required Libraries

The project requires a Python environment (3.x recommended) and the following external packages:

numpy>=1.20
matplotlib>=3.3


You can install them using the provided requirements.txt file:

pip install -r requirements.txt


2. Execution Order

The entire project must be executed sequentially: first training (train.py), then evaluation (eval.py).


Step 1: Training the Model

This step reads the data, splits it (60%/20%/20% sequentially), performs standardization, trains the Logistic Regression model using Stochastic Gradient Descent (SGD), and saves the trained parameters and data sets into the results/ folder.

python train.py

Expected Output: Console logs showing the loss convergence over epochs and confirmation of saved files (final_theta.npy, loss_history.png, etc.).


Step 2: Evaluating the Model

This step loads the saved parameters and the test/validation data from the results/ folder, calculates the confusion matrix components (TP, TN, FP, FN), and computes the final required metrics (accuracy, precision, recall, f1-score) for the train, validation, and test sets.

python eval.py

Expected Output: Console output displaying the calculated metrics and a confirmation that the final results are saved in results/metrics_results.txt.


IV. Core Implementation Details

The following functions were manually implemented without using built-in ML methods:

sigmoid(z): The activation function g(z) = 1 / (1 + e^{-z})

compute_cross_entropy_loss(...): The loss function used for training.

update_weights_sgd(...): Updates weights based on the gradient calculated for a single sample (SGD).
