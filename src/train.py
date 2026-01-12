import pandas as pd
import torch
import dgl
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import classification_report
from gnn import GNN
from graph_dataset import Dataset
import argparse
from sklearn.metrics import precision_recall_curve, auc, matthews_corrcoef
from metrics import plot_pr_curve, evaluate_predictions
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 4
NUM_FOLDS = 5
IN_DIM = 85
HIDDEN_DIM = 104
N_CLASSES = 2
N_CONVS = 4





def train(train_dataloader,valid_dataloader,epochs,model,leakage):
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        num_epochs = epochs
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs, gamma=0.5
        )
        loss_fcn =nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for batched_g, labels, names in train_dataloader:

                batched_g, labels = batched_g.to(DEVICE), labels[:, leakage].long().to(DEVICE)
                logits = model(
                    batched_g, batched_g.ndata["features"])
                loss = loss_fcn(logits, labels)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            avg_loss = total_loss / len(train_dataloader)
            train_acc = evaluate(train_dataloader, model, leakage,evaluation_report=False)
            val_acc = evaluate(valid_dataloader, model, leakage,evaluation_report=False)
            print(
                f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}"
            )
        val_acc, class_report = evaluate(valid_dataloader, model, leakage ,evaluation_report=True)
        return val_acc,class_report


def cross_validate(dataset, leakage,epochs,batch_size):
    Y = dataset.label[:, leakage]
    X = np.arange(len(Y))

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    accuracies = []
    overall_class_reports = {
        "0.0": {"precision": [], "recall": [], "f1-score": []},
        "1.0": {"precision": [], "recall": [], "f1-score": []},
        "mcc" : {"mcc":[]},
        "pr_auc": {"pr_auc": []}
    }

    print(batch_size)
    for fold, (train_indices, val_indices) in enumerate(skf.split(X, Y)):


        print(f"{'-' * 20} Fold {fold + 1}: Num train indices: {len(train_indices)}, "
              f"Num val indices: {len(val_indices)} {'-' * 20}")

        train_dataloader = GraphDataLoader(
            dataset,
            sampler=SubsetRandomSampler(train_indices),
            batch_size=batch_size,
            worker_init_fn=lambda _: np.random.seed(42)
        )
        valid_dataloader = GraphDataLoader(
            dataset,
            sampler=SubsetRandomSampler(val_indices),
            batch_size=batch_size,
            worker_init_fn=lambda _: np.random.seed(42)
        )


        model = GNN(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES,n_convs=N_CONVS).to(DEVICE)
        val_acc,class_report=train(train_dataloader,valid_dataloader,epochs,model,leakage)

        accuracies.append(val_acc)
        for metric in overall_class_reports.keys():
            if metric in class_report:
                for stat in overall_class_reports[metric].keys():
                    overall_class_reports[metric][stat].append(class_report[metric][stat])

    averaged_class_reports = {}
    for metric, stats in overall_class_reports.items():
        averaged_class_reports[metric] = {stat: np.mean(values) for stat, values in stats.items()}

    average_accuracies = np.mean(accuracies)

    print("-" * 44 + " Overall results" + "-" * 44 + "\n")
    print("\nCross-Validation Results:")
    print(f"Average accuracy Loss: {average_accuracies:.4f}")
    print("Average Class Reports:")
    for metric, stats in averaged_class_reports.items():
        print(f"{metric}: {stats}")



    print("\n\n\nTraining on the entire dataset....")
    entire_data_dataloader = GraphDataLoader(
            dataset,
            sampler=SubsetRandomSampler(range(len(dataset))),
            batch_size=batch_size,
            worker_init_fn=lambda _: np.random.seed(42)
        )
    val_acc,class_report=train(entire_data_dataloader,entire_data_dataloader,epochs,model,leakage)
    print("\nSaving the pre-trained model ....")
    print("Model saved.")
    torch.save(model.state_dict(), f"models/{'preprocessing_' if leakage ==1 else 'overlap_'}classifier.pth")







@torch.no_grad()
def evaluate(dataloader, model, leakage, evaluation_report=False):
    model.eval()
    total = 0
    total_correct = 0
    total_predictions = []
    total_labels = []
    total_probs = []  # Store probabilities for AUC calculation
    name = []

    for batched_graph, labels, names in dataloader:

        name.extend(names)
        batched_graph = batched_graph.to(DEVICE)
        labels = labels[:, leakage].to(DEVICE).to(torch.float32).flatten()
        feat = batched_graph.ndata.pop("features")
        total += len(labels)
        logits = model(batched_graph, feat)

        # Extract predictions and probabilities
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probabilities for the positive class
        threshold = 0.4  # Adjust to control precision/recall
        predicted = (probs >= threshold).to(torch.float32)

        total_predictions.append(predicted)
        total_labels.append(labels)
        total_probs.append(probs)  # Store probabilities

        total_correct += (predicted == labels).sum().item()

    # Compute overall accuracy
    accuracy = 1.0 * total_correct / total

    # Concatenate all predictions, labels, and probabilities
    total_predictions = torch.cat(total_predictions, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    total_probs = torch.cat(total_probs, dim=0)

    pr_auc = None
    mcc = None
    if len(torch.unique(total_labels)) > 1:  # Ensure there are both classes for AUC calculation
        precision, recall, _ = precision_recall_curve(total_labels.cpu().numpy(), total_probs.cpu().numpy())
        pr_auc = auc(recall, precision)
        mcc = matthews_corrcoef(total_labels.cpu().numpy(), total_predictions.cpu().numpy())



    if evaluation_report:
        # Generate classification report and confusion matrix
        class_report, conf_matrix, missed_cases, class_report_dict = evaluate_predictions(
            total_predictions.cpu(), total_labels.cpu(), name
        )
        print("Classification Report:")
        print(class_report)
        # Add ROC-AUC to the classification report dictionary
        if pr_auc is not None:
            class_report_dict["pr_auc"] = {"pr_auc":pr_auc}
        if mcc is not None:
            class_report_dict["mcc"] = {"mcc":mcc}
       
        return accuracy, class_report_dict

    return accuracy



def main():
    parser = argparse.ArgumentParser(description="Train the leakage classifier on a dataset of contextualized dependency graphs")
    parser.add_argument('--classifier', choices=['preprocessing', 'overlap'], required=True, help="Choose which classifier to train. Preprocessing Leakage or Overlap Leakage classifier.")
    parser.add_argument('--data', required=True,help="Path to the data folder or the bin file where the graphs are located.")
    parser.add_argument('--batch-size', default=512,help="Batch size of the dataloaders")
    parser.add_argument('--annotation',  required=True, help="Path to the CSV file where the ground truths are stored.")
    

    args = parser.parse_args()
    leakage = 1 if args.classifier == "preprocessing" else 0
    dataset = Dataset(dataset_path=args.annotation, data_path=args.data)
    cross_validate(dataset, leakage,NUM_EPOCHS,int(args.batch_size))

if __name__ == "__main__":
    main()

