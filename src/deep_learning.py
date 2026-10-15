"""
Deep learning baseline for Phase 8: an embedding MLP (categorical features
via learned embeddings, concatenated with scaled numeric features, then a
small feed-forward network). This is the standard modern DL architecture for
tabular data (Guo & Berkhahn, "Entity Embeddings of Categorical Variables",
2016) -- used here instead of PERI_DEP's original RNN-LSTM because that
architecture required treating PHQ-9 items as a sequence, and those items
are exactly what Phase 2's leakage audit excluded. An embedding MLP is the
correct DL architecture for the SDoH-only tabular feature set this project
actually uses.

NOTE ON TESTING: this module was NOT execution-tested (this environment has
no torch installed and no internet access to add it). The API usage follows
standard, stable PyTorch patterns, but if anything errors, send the full
traceback rather than trying to debug blind -- I'll fix it directly.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TabularDataset(Dataset):
    def __init__(self, numeric_array, categorical_array, labels):
        self.numeric = torch.tensor(numeric_array, dtype=torch.float32)
        self.categorical = torch.tensor(categorical_array, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.numeric[idx], self.categorical[idx], self.labels[idx]


class EmbeddingMLP(nn.Module):
    """
    One embedding layer per categorical column, concatenated with numeric
    features, fed through a small MLP with BatchNorm + Dropout.
    embedding_cardinalities: list of (num_categories, embedding_dim) tuples,
    one per categorical column, in the same column order used to build the
    categorical tensor.
    """
    def __init__(self, n_numeric, embedding_cardinalities, hidden_dims=(64, 32), dropout=0.3):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in embedding_cardinalities
        ])
        total_emb_dim = sum(emb_dim for _, emb_dim in embedding_cardinalities)
        input_dim = n_numeric + total_emb_dim

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_numeric, x_categorical):
        embedded = [emb(x_categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_numeric] + embedded, dim=1) if embedded else x_numeric
        return self.mlp(x).squeeze(1)


def embedding_dim_heuristic(cardinality):
    """Standard rule of thumb (fastai/Guo & Berkhahn): min(50, (cardinality+1)//2)."""
    return min(50, max(2, (cardinality + 1) // 2))


def prepare_tabular_arrays(train_df, cal_df, test_df, numeric_cols, categorical_cols):
    """
    Fit preprocessing on train only, apply to all three partitions.
    Returns numeric arrays (scaled), categorical arrays (ordinal-encoded,
    with an extra "unseen" index reserved), and embedding_cardinalities
    ready for EmbeddingMLP.
    """
    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    num_train = scaler.fit_transform(num_imputer.fit_transform(train_df[numeric_cols]))
    num_cal = scaler.transform(num_imputer.transform(cal_df[numeric_cols]))
    num_test = scaler.transform(num_imputer.transform(test_df[numeric_cols]))

    cat_imputer = SimpleImputer(strategy="most_frequent")
    train_cat_filled = cat_imputer.fit_transform(train_df[categorical_cols])
    cal_cat_filled = cat_imputer.transform(cal_df[categorical_cols])
    test_cat_filled = cat_imputer.transform(test_df[categorical_cols])

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_train = encoder.fit_transform(train_cat_filled)
    cat_cal = encoder.transform(cal_cat_filled)
    cat_test = encoder.transform(test_cat_filled)

    # Reserve index 0 for "unseen" categories (present in cal/test but not train).
    # OrdinalEncoder gives unseen categories code -1, known categories 0..k-1.
    # Adding 1 uniformly maps unseen -1 -> 0 (the reserved slot) and known
    # 0..k-1 -> 1..k, with no collision and no special-casing needed.
    cat_train = cat_train + 1
    cat_cal = cat_cal + 1
    cat_test = cat_test + 1

    embedding_cardinalities = []
    for i, col in enumerate(categorical_cols):
        # cat_train[:, i] ranges 1..k after the +1 shift (0 is reserved for unseen,
        # which train never contains but cal/test might); embedding table needs to
        # cover 0..k inclusive, i.e. k+1 slots = cat_train[:, i].max() + 1.
        n_categories = int(cat_train[:, i].max()) + 1
        emb_dim = embedding_dim_heuristic(n_categories)
        embedding_cardinalities.append((n_categories, emb_dim))

    return (num_train, cat_train.astype(int), num_cal, cat_cal.astype(int),
            num_test, cat_test.astype(int), embedding_cardinalities)


def train_mlp(model, train_loader, cal_dataset, device, y_train, seed,
              max_epochs=100, patience=10, lr=1e-3, weight_decay=1e-4):
    """
    Train with class-weighted BCE loss (handles imbalance without needing
    SMOTE -- SMOTE's linear interpolation doesn't cleanly apply to embedding-
    indexed categorical columns, so a weighted loss is the standard DL
    alternative for imbalanced tabular classification). Early-stops on
    calibration-set AUC, matching how every other model in this project was
    validated.
    """
    model.to(device)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    cal_numeric, cal_categorical, cal_labels = cal_dataset.numeric.to(device), cal_dataset.categorical.to(device), cal_dataset.labels.numpy()

    best_auc = -1
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_num, x_cat)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            cal_logits = model(cal_numeric, cal_categorical)
            cal_proba = torch.sigmoid(cal_logits).cpu().numpy()
        cal_auc = roc_auc_score(cal_labels, cal_proba)

        if cal_auc > best_auc:
            best_auc = cal_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_auc


def predict_proba_mlp(model, numeric_array, categorical_array, device):
    model.eval()
    with torch.no_grad():
        x_num = torch.tensor(numeric_array, dtype=torch.float32).to(device)
        x_cat = torch.tensor(categorical_array, dtype=torch.long).to(device)
        logits = model(x_num, x_cat)
        return torch.sigmoid(logits).cpu().numpy()
