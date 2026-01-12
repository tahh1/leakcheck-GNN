from sklearn.metrics import precision_recall_curve, auc, confusion_matrix,classification_report
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_pr_curve(true_labels, predicted_probs, title="Precision-Recall Curve"):
    precision, recall, thresholds = precision_recall_curve(true_labels, predicted_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f"PR Curve (area = {pr_auc:.2f})")

    # Annotate all thresholds as in the original function
    for i, thresh in enumerate(thresholds):
        if i % (len(thresholds) // 10) == 0:  # Reduce clutter, keep key points
            plt.annotate(f"{thresh:.2f}",
                         (recall[i], precision[i]),
                         textcoords="offset points", xytext=(5,5),
                         ha='left', fontsize=8, color='gray')

    # Find and highlight the key thresholds
    if len(thresholds) > 0:
        # Threshold = 0.5 (Default decision boundary)
        closest_0_5_idx = np.argmin(np.abs(thresholds - 0.5))
        plt.scatter(recall[closest_0_5_idx], precision[closest_0_5_idx], color='blue', s=50, label="Threshold 0.5")
        plt.annotate(f"0.5", (recall[closest_0_5_idx], precision[closest_0_5_idx]),
                     textcoords="offset points", xytext=(5,5), ha='left', fontsize=9, color='blue')

        # Threshold for Recall ≈ 80%
        recall_80_idx = np.argmin(np.abs(recall - 0.8))
        plt.scatter(recall[recall_80_idx], precision[recall_80_idx], color='red', s=50, label="Recall ≈ 80%")
        plt.annotate(f"{thresholds[recall_80_idx]:.2f}", (recall[recall_80_idx], precision[recall_80_idx]),
                     textcoords="offset points", xytext=(5,5), ha='left', fontsize=9, color='red')

        # Threshold for Precision ≈ 80%
        precision_80_idx = np.argmin(np.abs(precision - 0.8))
        plt.scatter(recall[precision_80_idx], precision[precision_80_idx], color='green', s=50, label="Precision ≈ 80%")
        plt.annotate(f"{thresholds[precision_80_idx]:.2f}", (recall[precision_80_idx], precision[precision_80_idx]),
                     textcoords="offset points", xytext=(5,5), ha='left', fontsize=9, color='green')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()





def evaluate_predictions(predictions, true_labels,graph_names):
    # Generate classification report
    ones_tensor = torch.ones_like(predictions)

    class_report_dict = classification_report(true_labels,predictions, output_dict=True)
    class_report = classification_report(true_labels,predictions)



    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Find indices of false positives and false negatives
    false_positives = np.where((predictions[:] == 1) & (true_labels[:] == 0))[0]
    false_negatives = np.where((predictions[:] == 0) & (true_labels[:] == 1))[0]
    true_positives = np.where((predictions[:] == 1) & (true_labels[:] == 1))[0]

    missed_cases = {
        "false_positives": [graph_names[i] for i in false_positives.tolist()],
        "false_negatives": [graph_names[i] for i in false_negatives.tolist()],
        "true_positives": [graph_names[i] for i in true_positives.tolist()],

    }

    return class_report, conf_matrix, missed_cases, class_report_dict