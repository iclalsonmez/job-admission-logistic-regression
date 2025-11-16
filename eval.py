import numpy as np
import os
import sys

RESULTS_PATH = "results"
THRESHOLD = 0.50 # classification threshold

# main functions
def sigmoid(z):
    """Applies Sigmoid activation function."""
    z = np.clip(z, -500, 500) 
    return 1.0 / (1.0 + np.exp(-z))

def predict(X, theta, threshold=THRESHOLD):
    """
    Calculates probabilities using the trained parameters and makes the class prediction.
    """
    # hypothesis: probability = sigmoid(X * theta)
    probabilities = sigmoid(X @ theta)
    # classification: if p>0.50 -> 1, else 0 
    return (probabilities >= threshold).astype(int)

# metric calculation functions  
def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculates TP, TN, FP, FN values for confusion matrix.
    """
    TP = np.sum((y_true == 1) & (y_pred == 1)) # TP
    TN = np.sum((y_true == 0) & (y_pred == 0)) # TN
    FP = np.sum((y_true == 0) & (y_pred == 1)) # FP
    FN = np.sum((y_true == 1) & (y_pred == 0)) # FN
    return TP, TN, FP, FN

def calculate_metrics(TP, TN, FP, FN):
    """
    Calculates accuracy, precision, recall and F1-score using TP, TN, FP, FN.
    """
    total = TP + TN + FP + FN
    
    # control denominators to avoid division by zero
    precision_den = TP + FP
    recall_den = TP + FN
    f1_den = 0 
    
    # accuracy
    accuracy = (TP + TN) / total if total > 0 else 0
    
    # precision
    precision = TP / precision_den if precision_den > 0 else 0
    
    # recall
    recall = TP / recall_den if recall_den > 0 else 0
    
    # f1-score 
    f1_den = precision + recall
    f1_score = (2 * precision * recall) / f1_den if f1_den > 0 else 0
    
    return accuracy, precision, recall, f1_score


# evaluation functions
def load_evaluation_data():
    """
    Loads the trained parameters and datasets from the results/ folder.
    """
    try:
        final_theta = np.load(os.path.join(RESULTS_PATH, 'final_theta.npy'))
        
        # loading train, validation and test data
        data_sets = {
            "Train": (np.load(os.path.join(RESULTS_PATH, 'X_train.npy')), np.load(os.path.join(RESULTS_PATH, 'y_train.npy'))),
            "Validation": (np.load(os.path.join(RESULTS_PATH, 'X_val.npy')), np.load(os.path.join(RESULTS_PATH, 'y_val.npy'))),
            "Test": (np.load(os.path.join(RESULTS_PATH, 'X_test.npy')), np.load(os.path.join(RESULTS_PATH, 'y_test.npy')))
        }
        
        return final_theta, data_sets
        
    except FileNotFoundError as e:
        print(f"ERROR: The necessary files for evaluation were not found. Please run the train.py file first.")
        print(f"Missing file: {e.filename}")
        sys.exit(1)


def evaluate_model():
    """
    Calculates and saves the metrics for all datasets.
    """
    final_theta, data_sets = load_evaluation_data()
        
    print("--- Model Evaluation Metrics ---")
    
    results = {}
    
    for name, (X, y_true) in data_sets.items():
        # prediction
        y_pred = predict(X, final_theta)
        
        # calculating confusion matrix
        TP, TN, FP, FN = calculate_confusion_matrix(y_true, y_pred)
        
        # calculating metrics
        accuracy, precision, recall, f1_score = calculate_metrics(TP, TN, FP, FN)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
        }
        
        print(f"\n{name} Results (Total examples: {len(y_true)}):")
        print(f"  > Accuracy: {accuracy:.4f}")
        print(f"  > Precision: {precision:.4f}")
        print(f"  > Recall: {recall:.4f}")
        print(f"  > F1-Score: {f1_score:.4f}")
        
    # saving results into .txt file
    results_file = os.path.join(RESULTS_PATH, 'metrics_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("LR Evaluation Metrics:\n")
        f.write("-------------------------------------------\n")
        for name, res in results.items():
            f.write(f"\n{name} :\n")
            f.write(f"Accuracy: {res['Accuracy']:.4f}\n")
            f.write(f"Precision: {res['Precision']:.4f}\n")
            f.write(f"Recall: {res['Recall']:.4f}\n")
            f.write(f"F1-Score: {res['F1-Score']:.4f}\n")
            f.write(f"TP={res['TP']}, TN={res['TN']}, FP={res['FP']}, FN={res['FN']}\n")
            
    print(f"\nAll metrics results saved into results/metrics_results.txt.")

if __name__ == "__main__":
    evaluate_model()