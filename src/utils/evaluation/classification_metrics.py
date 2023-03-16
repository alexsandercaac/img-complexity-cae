"""
    Module to implement functions that calculate classification metrics.
"""
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score


def get_classification_metrics(y_true, y_pred):
    """
    Calculates the classification metrics.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.

    Returns:
        A dictionary with the classification metrics.
    """
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return {'f1': f1, 'accuracy': accuracy, 'precision': precision,
            'recall': recall}
