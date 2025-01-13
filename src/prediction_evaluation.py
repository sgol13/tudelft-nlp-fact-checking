import evaluate
from typing import List, Tuple
from sklearn.metrics import classification_report, confusion_matrix,f1_score

from src.common import QTDataset, QT_VERACITY_LABELS
from src.quantemp_processor import qt_veracity_label_encoder

_CATEGORIES = ["statistical", "temporal", "interval", "comparison"]

_f1_metric = evaluate.load("f1")

def _filter_category(claims: QTDataset, predictions: List[int], category: str) -> Tuple[QTDataset, List[int]]:
    filtered_claims = []
    filtered_predictions = []

    for claim, prediction in zip(claims, predictions):
        taxonomy_label = claim["taxonomy_label"].strip()
        assert taxonomy_label in _CATEGORIES, f"Unknown category: {taxonomy_label}"

        if taxonomy_label == category:
            filtered_claims.append(claim)
            filtered_predictions.append(prediction)

    return filtered_claims, filtered_predictions


def _calculate_macro_weighted_f1(claims: QTDataset, predictions: List[int]) -> Tuple[float, float]:
    gt_labels = [claim["label"] for claim in claims]
    gt_labels = qt_veracity_label_encoder.transform(gt_labels)
    macro_f1 = _f1_metric.compute(references=gt_labels, predictions=predictions, average="macro")['f1']
    weighted_f1 = _f1_metric.compute(references=gt_labels, predictions=predictions, average="weighted")['f1']
    return macro_f1, weighted_f1


def evaluate_predictions(claims: QTDataset, predictions: List[int]):
    gt_labels = [claim["label"] for claim in claims]
    gt_labels = qt_veracity_label_encoder.transform(gt_labels)

    print(classification_report(gt_labels, predictions, target_names=QT_VERACITY_LABELS, digits=4))
    print(confusion_matrix(gt_labels, predictions))

    table_row = []

    # per category
    for category in _CATEGORIES:
        filtered_claims, filtered_predictions = _filter_category(claims, predictions, category)
        macro_f1, weighted_f1 = _calculate_macro_weighted_f1(filtered_claims, filtered_predictions)
        table_row.extend([macro_f1, weighted_f1])
        print(f'{category}: {macro_f1:.4f} {weighted_f1:.4f}')

    # Per-class F1
    cr = classification_report(gt_labels, predictions, target_names=QT_VERACITY_LABELS, digits=4, output_dict=True)
    f1_scores_per_class = {label: metrics['f1-score'] for label, metrics in cr.items() if isinstance(metrics, dict)}

    t_f1 = f1_scores_per_class["True"]
    f_f1 = f1_scores_per_class["False"]
    c_f1 = f1_scores_per_class["Conflicting"]
    table_row.extend([t_f1, f_f1, c_f1])

    # Quantemp
    macro_f1 = _f1_metric.compute(references=gt_labels, predictions=predictions, average="macro")['f1']
    weighted_f1 = _f1_metric.compute(references=gt_labels, predictions=predictions, average="weighted")['f1']
    table_row.extend([macro_f1, weighted_f1])

    print()
    for cat in _CATEGORIES + ['per-class', 'QuanTemp']:
        print(cat, end=' ')
    print()
    for num in table_row:
        print(f'{100 * num:.2f}', end=' ')
    print()