from sklearn.metrics import accuracy_score, classification_report

def display_performance_metrics(true_labels_mi, predicted_labels_mi, true_labels_pg, predicted_labels_pg, model_name):
    """
    Calculate and display performance metrics for both mistake identification and guidance.
    
    Args:
        true_labels_mi: Ground truth labels for mistake identification
        predicted_labels_mi: Predicted labels for mistake identification
        true_labels_pg: Ground truth labels for providing guidance
        predicted_labels_pg: Predicted labels for providing guidance
        model_name: Name of the model for display purposes
    """
    print(f"\n--- {model_name} Model Performance Metrics ---\n")

    print("--- Mistake Identification ---")
    mi_accuracy = accuracy_score(true_labels_mi, predicted_labels_mi)
    print(f"Accuracy: {mi_accuracy:.2f}")
    print(classification_report(true_labels_mi, predicted_labels_mi, zero_division=0))

    print("\n" + "="*50 + "\n")

    print("--- Providing Guidance ---")
    pg_accuracy = accuracy_score(true_labels_pg, predicted_labels_pg)
    print(f"Accuracy: {pg_accuracy:.2f}")
    print(classification_report(true_labels_pg, predicted_labels_pg, zero_division=0))
