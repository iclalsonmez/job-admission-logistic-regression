import numpy as np
import matplotlib.pyplot as plt
import os
import math

# paths and hyperparameters
DATA_PATH = "dataset/hw1Data.txt"
RESULTS_PATH = "results"
ALPHA = 0.01  # learning rate
N_EPOCHS = 1000 # number of epochs
# THRESHOLD = 0.50 

# functions
def load_data(file_path):
    """
    Reads the file at the specified path (DATA_PATH) and converts it to a NumPy array
    Arguments:
        file_path - path to the data file ("dataset/hw1Data.txt")
    Returns:
        X - exam scores (features)
        y - acceptance/rejection labels (targets)
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(list(map(float, line.strip().split(','))))
    except FileNotFoundError:
        print(f"ERROR: Data file not found at: {file_path}.")
        return np.array([]), np.array([])
        
    data = np.array(data)
    if data.size == 0:
        return np.array([]), np.array([])
        
    X = data[:, :2] # input info: first two values
    y = data[:, 2]  # output info: last value
    return X, y

def preprocess_data(X):
    """
    Standardizes (normalizes) the data and adds the bias (intercept) term
    Returns:
        X_proc - prepared feature matrix (N x 3)
        mu - mean vector (stored for normalization)
        sigma - standard deviation vector (stored for normalization)
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1e-8 # error prevention for zero std
    
    X_norm = (X - mu) / sigma
    # for bias term (intercept)
    X_proc = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))
    return X_proc, mu, sigma

def split_data(X_proc, y):
    """
    Splits the data sequentially into 60% training, 20% validation, and 20% test sets
    """
    m = X_proc.shape[0]
    train_size = int(0.6 * m)
    val_size = int(0.2 * m)
    
    X_train, y_train = X_proc[:train_size], y[:train_size]
    X_val, y_val = X_proc[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X_proc[train_size + val_size:], y[train_size + val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# logistic regression functions
def sigmoid(z):
    """
    Sigmoid function is being used for activation function
    """
    z = np.clip(z, -500, 500) 
    return 1.0 / (1.0 + np.exp(-z))

def compute_cross_entropy_loss(y_target, y_predicted):
    """
    Calculates cross-entropy loss
    """
    epsilon = 1e-15 # log(0) error prevention
    y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
    
    # loss = -[y_target * log(y_predicted) + (1-y_target) * log(1-y_predicted)]
    if isinstance(y_target, (int, float)): 
        return -(y_target * math.log(y_predicted) + (1 - y_target) * math.log(1 - y_predicted))
    else: # batch
        return -(y_target * np.log(y_predicted) + (1 - y_target) * np.log(1 - y_predicted))

def update_weights_sgd(X_i, y_i, theta, alpha):
    """
    Updates the weights using stochastic gradient descent for a single training example
    """
    # hypothesis
    h = sigmoid(np.dot(X_i, theta))
    
    # error and gradient
    error = h - y_i
    gradient = error * X_i 
    
    # theta update
    theta = theta - alpha * gradient
    
    return theta

# training
def train_model(X_train, y_train, X_val, y_val, alpha, n_epochs):
    """
    Trains the logistic regression model using SGD
    Returns: final_theta, train_loss_history, val_loss_history
    """
    n_features = X_train.shape[1]
    theta = np.zeros(n_features) 
    
    train_loss_history = []
    val_loss_history = []
    
    print(f"Train starts... (Epoch: {n_epochs}, Alpha: {alpha})")
    
    for epoch in range(1, n_epochs + 1):
        epoch_train_losses = []
        
        # SGD over training examples
        for i in range(X_train.shape[0]):
            X_i = X_train[i]
            y_i = y_train[i]
            
            # weight update (SGD)
            theta = update_weights_sgd(X_i, y_i, theta, alpha)
            
            # calculating loss
            y_pred_i = sigmoid(np.dot(X_i, theta))
            loss_i = compute_cross_entropy_loss(y_i, y_pred_i)
            epoch_train_losses.append(loss_i)

        # average train loss
        avg_train_loss = np.mean(epoch_train_losses)
        train_loss_history.append(avg_train_loss)
        
        # validation loss (at the end of epoch, over validation set)
        val_predictions = sigmoid(X_val @ theta)
        avg_val_loss = np.mean(compute_cross_entropy_loss(y_val, val_predictions))
        val_loss_history.append(avg_val_loss)

        if epoch % 100 == 0 or epoch == n_epochs:
            print(f"Epoch {epoch}/{n_epochs} -> Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            
    print("Training completed.")
    return theta, train_loss_history, val_loss_history


def plot_loss_history(train_loss, val_loss): 
    """
    Plots train and validation loss according to epoch and saves into "results" folder.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    
    plt.title('Train and Validation Loss Evaluation')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Average Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    plot_filename = os.path.join(RESULTS_PATH, 'loss_history.png')
    plt.savefig(plot_filename)
    print(f"\nLoss graph saved: {plot_filename}") # for evaluating the overfitting
    
    
def plot_data_distribution(X, y):
    """
    Plots the class distribution for 1st Exam Score (x-axis) and 2nd Exam Score (y-axis)
    """
    plt.figure(figsize=(8, 6))
    accepted = X[y == 1]
    rejected = X[y == 0]

    plt.scatter(rejected[:, 0], rejected[:, 1], c='red', marker='x', label='Rejected (0)')
    plt.scatter(accepted[:, 0], accepted[:, 1], c='blue', marker='o', label='Accepted (1)')

    plt.xlabel('1. Exam Note (x1)')
    plt.ylabel('2. Exam Note (x2)')
    plt.title('Distribution of Job Application Results')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    plot_filename = os.path.join(RESULTS_PATH, 'data_distribution.png')
    plt.savefig(plot_filename)
    print(f"Graph saved: {plot_filename}")


# main
def main():
    # data loading
    X_raw, y = load_data(DATA_PATH)
    
    if X_raw.size == 0:
        return 

    # plotting class distribution
    plot_data_distribution(X_raw, y)

    # preprocessing (normalization and bias)
    X_proc, mu, sigma = preprocess_data(X_raw)
    
    # data splitting
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_proc, y)
    
    # model training with SGD
    final_theta, train_loss_history, val_loss_history = train_model(
        X_train, y_train, X_val, y_val, ALPHA, N_EPOCHS
    )
    
    # loss plotting
    plot_loss_history(train_loss_history, val_loss_history)
    
    # saving results
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
        
    np.save(os.path.join(RESULTS_PATH, 'final_theta.npy'), final_theta)
    np.save(os.path.join(RESULTS_PATH, 'mu.npy'), mu)
    np.save(os.path.join(RESULTS_PATH, 'sigma.npy'), sigma)

    # saving test data for eval.py
    np.save(os.path.join(RESULTS_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(RESULTS_PATH, 'y_test.npy'), y_test)
    np.save(os.path.join(RESULTS_PATH, 'X_val.npy'), X_val)
    np.save(os.path.join(RESULTS_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(RESULTS_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(RESULTS_PATH, 'y_train.npy'), y_train)
    
    print("\nTraining completed and results saved in 'results/' folder.")

if __name__ == "__main__":
    main()