import argparse
import torch
from sklearn.metrics import precision_recall_curve, auc, matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from metrics import evaluate_predictions
import pandas as pd
from graph_dataset import Dataset
from gnn import GNN
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 40
IN_DIM = 85
HIDDEN_DIM = 104
N_CLASSES = 2
N_CONVS = 4

@torch.no_grad()
def test(dataset, model_path, leakage, df,batch_size):
    print("Loading model ....")
    model = GNN(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES,n_convs=N_CONVS).to(DEVICE)    
    model.load_state_dict(torch.load(model_path,map_location=torch.device(DEVICE)))
    model.eval()

    dataloader = GraphDataLoader(
            dataset,
            sampler=SubsetRandomSampler(list(range(len(dataset)))),
            batch_size=batch_size,
            worker_init_fn=lambda _: np.random.seed(42)
        )


    total = 0
    total_correct = 0
    total_predictions = []
    total_labels = []
    total_probs = []  # Store probabilities for AUC calculation
    name = []
    leakage_name = "preproc" if leakage ==1 else "overlap"
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
        #predicted = torch.max(logits, 1)[1].to(torch.float32).to(device)

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
    pr_auc = None
    if len(torch.unique(total_labels)) > 1:  # Ensure there are both classes for AUC calculation
        precision, recall, _ = precision_recall_curve(total_labels.cpu().numpy(), total_probs.cpu().numpy())
        pr_auc = auc(recall, precision)
        mcc = matthews_corrcoef(total_labels.cpu().numpy(), total_predictions.cpu().numpy())



 
    # Generate classification report and confusion matrix
    class_report, conf_matrix, missed_cases, class_report_dict = evaluate_predictions(
            total_predictions.cpu(), total_labels.cpu(), name
    )
    print("-" * 44 + " Overall test results" + "-" * 44 + "\n")
    print("Classification Report:")
    print(class_report)


    # Add ROC-AUC to the classification report dictionary
    print(f"Accuracy: {accuracy}")
    if pr_auc is not None:
        print("pr_auc:  ",pr_auc)
    if mcc is not None:
        print("mcc:  ",mcc)

    df_ours = pd.DataFrame({"File":[n.split("/")[0] for n in name],"pair":[n.split("/")[1] for n in name],f"{leakage_name} ours":[int(x) for x in total_predictions.cpu().tolist()]})
    merged_df = pd.merge(
    df[['File', f'{leakage_name} GT',
        f'{leakage_name} pattern', 'pair']],
            df_ours,
            how='right',
            left_on=["File", "pair"],
            right_on=["File", "pair"]
            )
    performance_balanced = merged_df.groupby(f"{leakage_name} pattern").apply(
        lambda x: pd.Series({
        "Recall": recall_score(x[f"{leakage_name} GT"], x[f"{leakage_name} ours"]),
        })
        ).reset_index()
    print("\nPer-scenario model recall:")
    print(performance_balanced)







def main():
    parser = argparse.ArgumentParser(description="Test the pre-trained model on a test data and show granular per-scenario recall.")
    parser.add_argument('--model', required=True, help="The model path")
    parser.add_argument('--classifier', choices=['preprocessing', 'overlap'], required=True, help="Choose which classifier to train. Preprocessing Leakage or Overlap Leakage classifier.")
    parser.add_argument('--data', required=True,help="Path to the data folder or the bin file where the graphs are located.")
    parser.add_argument('--batch-size', default=512, help="Batch size of the dataloaders")
    parser.add_argument('--annotation',  required=True, help="Path to the CSV file where the ground truths are stored.")
    args = parser.parse_args()

    leakage = 1 if args.classifier == "preprocessing" else 0
    print("Loading the dataset ....")
    dataset = Dataset(dataset_path=args.annotation, data_path=args.data)
    df = pd.read_csv(args.annotation)
    test(dataset, args.model, leakage, df, int(args.batch_size))


if __name__ == "__main__":
    main()