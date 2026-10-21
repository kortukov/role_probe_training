"""Token accuracy and conversation-balanced mean role probabilities."""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss

from .harmony import ROLES


def metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    present = np.unique(labels)
    return {
        "n_tokens": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "nll": float(log_loss(labels, probabilities, labels=np.arange(len(ROLES)))),
        "uniform_accuracy": 1 / len(ROLES),
        "majority_accuracy": float(np.bincount(labels, minlength=len(ROLES)).max() / len(labels)),
        "confusion_counts": confusion_matrix(labels, predictions, labels=np.arange(len(ROLES))).tolist(),
        "by_role": {ROLES[r]: {
            "n_tokens": int((labels == r).sum()),
            "accuracy": float((predictions[labels == r] == r).mean()),
            "mean_probability": float(probabilities[labels == r, r].mean()),
        } for r in present},
    }


def conversation_summary(rows):
    """First average tokens within (conversation, true role), then conversations.

    rows: iterable of {conversation_id, labels, probabilities}. No role/token
    balancing is silently applied to the overall token accuracy.
    """
    by_role = {}
    for role_id, role in enumerate(ROLES):
        means, accuracies, count = [], [], 0
        for row in rows:
            mask = row['labels'] == role_id
            if mask.any():
                p = row['probabilities'][mask]
                means.append(p.mean(axis=0))
                accuracies.append((p.argmax(axis=1) == role_id).mean())
                count += int(mask.sum())
        if means:
            by_role[role] = {
                'n_conversations': len(means), 'n_tokens': count,
                'mean_probabilities': np.mean(means, axis=0).tolist(),
                'accuracy': float(np.mean(accuracies)),
            }
    return by_role


def plot_summary(summary, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns = [r for r in ROLES if r in summary]
    data = np.asarray([summary[r]['mean_probabilities'] for r in columns]).T
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap='YlOrBr')
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(ROLES)), ROLES)
    ax.set_xlabel('Ground-truth message role')
    ax.set_ylabel('Predicted role probability')
    ax.set_title(title)
    for i in range(len(ROLES)):
        for j in range(len(columns)):
            ax.text(j, i, f'{data[i, j]:.1%}', ha='center', va='center',
                    color='white' if data[i, j] > .65 else 'black')
    fig.colorbar(im, ax=ax, label='Mean probability')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
