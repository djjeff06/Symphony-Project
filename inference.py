import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model_configs import CONFIG
from model import SymphonyClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

def compute_tp_tn_fp_fn(y_true, y_pred, num_classes):
    results = {}
    for c in range(num_classes):
        TP = np.sum((y_pred == c) & (y_true == c))
        FP = np.sum((y_pred == c) & (y_true != c))
        FN = np.sum((y_pred != c) & (y_true == c))
        TN = np.sum((y_pred != c) & (y_true != c))
        results[c] = {"TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)}
    return results

def save_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(8, 8))
    disp.plot(xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def inference(folder_path, model_name, mode):
    test_data = np.load(os.path.join(folder_path, "test.npz"))
    X_test = test_data["X"]
    y_composer_test = test_data["y_composer"]
    y_era_test = test_data["y_era"]

    X_tensor = torch.from_numpy(X_test).float()
    composer_tensor = torch.from_numpy(y_composer_test).long()
    era_tensor = torch.from_numpy(y_era_test).long()

    test_dataset = TensorDataset(X_tensor, composer_tensor, era_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SymphonyClassifier(
        input_size=X_test.shape[2],
        n_embedding=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"]
    ).to(device)

    model.load_state_dict(torch.load(model_name + ".pth", map_location=device))
    model.eval()

    all_composer_preds, all_composer_true = [], []
    all_era_preds, all_era_true = [], []
    correct_composer = correct_era = 0
    total_composer = total_era = 0

    with torch.no_grad():
        for Xb, composer_labels, era_labels in test_loader:
            Xb = Xb.to(device)
            composer_labels = composer_labels.to(device)
            era_labels = era_labels.to(device)

            composer_output, era_output = None, None
            if mode == "composer_era":
                composer_output, era_output = model.forward_composer_era(Xb, device=device)
            elif mode == "composer":
                composer_output = model.forward_composer(Xb, device=device)
            elif mode == "era":
                era_output = model.forward_era(Xb, device=device)

            if composer_output is not None:
                _, pred_c = torch.max(composer_output, 1)
                correct_composer += (pred_c == composer_labels).sum().item()
                total_composer += composer_labels.size(0)
                all_composer_preds.extend(pred_c.cpu().numpy())
                all_composer_true.extend(composer_labels.cpu().numpy())

            if era_output is not None:
                _, pred_e = torch.max(era_output, 1)
                correct_era += (pred_e == era_labels).sum().item()
                total_era += era_labels.size(0)
                all_era_preds.extend(pred_e.cpu().numpy())
                all_era_true.extend(era_labels.cpu().numpy())

    if total_composer > 0:
        composer_acc = 100 * correct_composer / total_composer
        print(f"Test Composer Accuracy: {composer_acc:.2f}%")
    if total_era > 0:
        era_acc = 100 * correct_era / total_era
        print(f"Test Era Accuracy: {era_acc:.2f}%")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    if total_composer > 0:
        save_confusion_matrix(
            np.array(all_composer_true),
            np.array(all_composer_preds),
            class_names=[str(i) for i in range(max(y_composer_test)+1)],
            title="Composer Confusion Matrix",
            filename=os.path.join(results_dir, "composer_confusion_matrix.png")
        )

    if total_era > 0:
        save_confusion_matrix(
            np.array(all_era_true),
            np.array(all_era_preds),
            class_names=[str(i) for i in range(max(y_era_test)+1)],
            title="Era Confusion Matrix",
            filename=os.path.join(results_dir, "era_confusion_matrix.png")
        )

    if total_composer > 0:
        print("\nComposer Task - Per Class TP/TN/FP/FN")
        stats = compute_tp_tn_fp_fn(np.array(all_composer_true), np.array(all_composer_preds), max(y_composer_test)+1)
        for c, s in stats.items():
            print(f"Class {c}: TP={s['TP']}  TN={s['TN']}  FP={s['FP']}  FN={s['FN']}")

    if total_era > 0:
        print("\nEra Task - Per Class TP/TN/FP/FN")
        stats = compute_tp_tn_fp_fn(np.array(all_era_true), np.array(all_era_preds), max(y_era_test)+1)
        for c, s in stats.items():
            print(f"Class {c}: TP={s['TP']}  TN={s['TN']}  FP={s['FP']}  FN={s['FN']}")

    if total_composer > 0:
        y_true_c = np.array(all_composer_true)
        y_pred_c = np.array(all_composer_preds)
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true_c, y_pred_c, average='micro', zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_c, y_pred_c, average='macro', zero_division=0)
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true_c, y_pred_c, average='weighted', zero_division=0)
        print("\nComposer Task - Overall Metrics")
        print(f"Micro: Precision={micro_p:.4f}, Recall={micro_r:.4f}, F1={micro_f1:.4f}")
        print(f"Macro: Precision={macro_p:.4f}, Recall={macro_r:.4f}, F1={macro_f1:.4f}")
        print(f"Weighted: Precision={weighted_p:.4f}, Recall={weighted_r:.4f}, F1={weighted_f1:.4f}")

    if total_era > 0:
        y_true_e = np.array(all_era_true)
        y_pred_e = np.array(all_era_preds)
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true_e, y_pred_e, average='micro', zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_e, y_pred_e, average='macro', zero_division=0)
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true_e, y_pred_e, average='weighted', zero_division=0)
        print("\nEra Task - Overall Metrics")
        print(f"Micro: Precision={micro_p:.4f}, Recall={micro_r:.4f}, F1={micro_f1:.4f}")
        print(f"Macro: Precision={macro_p:.4f}, Recall={macro_r:.4f}, F1={macro_f1:.4f}")
        print(f"Weighted: Precision={weighted_p:.4f}, Recall={weighted_r:.4f}, F1={weighted_f1:.4f}")

    print("\nAll confusion matrices saved in 'results/' folder.")


if __name__ == "__main__":
    if not len(sys.argv) == 4:
        raise Exception("Usage: python inference.py <data_path> <model_name> <mode>")
    inference(sys.argv[1], sys.argv[2], sys.argv[3])
