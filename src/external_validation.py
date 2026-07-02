"""External validation: train on cohort 1, test on cohort 2.

Reports AUC, sensitivity, specificity, FNR with bootstrap CIs.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm


def _fit_predict(X_train, y_train, X_test, classifier="logistic"):
    """Fit classifier on train, return predicted probabilities on test."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if classifier == "logistic":
        clf = LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs"
        )
    elif classifier == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss", verbosity=0
        )
    elif classifier == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    clf.fit(X_train_s, y_train)
    return clf.predict_proba(X_test_s)[:, 1]


def _metrics_from_probs(y_true, y_prob):
    """Compute AUC, sensitivity, specificity, FNR at Youden threshold."""
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    threshold = thresholds[best_idx]
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {"auc": auc, "sensitivity": sens, "specificity": spec, "fnr": fnr}


def external_validation(X1, y1, X2, y2, classifier="logistic", n_bootstrap=1000, seed=42):
    """Train on cohort 1, test on cohort 2 (and vice versa).

    Returns dict with forward/reverse metrics and bootstrap CIs.
    """
    rng = np.random.default_rng(seed)

    # Forward: train C1 -> test C2
    prob_fwd = _fit_predict(X1, y1, X2, classifier)
    metrics_fwd = _metrics_from_probs(y2, prob_fwd)

    # Reverse: train C2 -> test C1
    prob_rev = _fit_predict(X2, y2, X1, classifier)
    metrics_rev = _metrics_from_probs(y1, prob_rev)

    # Bootstrap CIs for forward AUC
    boot_aucs_fwd = []
    for _ in tqdm(range(n_bootstrap), desc="  bootstrap forward"):
        idx = rng.choice(len(y2), size=len(y2), replace=True)
        if len(np.unique(y2[idx])) < 2:
            continue
        boot_aucs_fwd.append(roc_auc_score(y2[idx], prob_fwd[idx]))

    boot_aucs_rev = []
    for _ in tqdm(range(n_bootstrap), desc="  bootstrap reverse"):
        idx = rng.choice(len(y1), size=len(y1), replace=True)
        if len(np.unique(y1[idx])) < 2:
            continue
        boot_aucs_rev.append(roc_auc_score(y1[idx], prob_rev[idx]))

    return {
        "forward": {
            **metrics_fwd,
            "auc_ci_95": (
                float(np.percentile(boot_aucs_fwd, 2.5)),
                float(np.percentile(boot_aucs_fwd, 97.5)),
            ) if boot_aucs_fwd else (np.nan, np.nan),
        },
        "reverse": {
            **metrics_rev,
            "auc_ci_95": (
                float(np.percentile(boot_aucs_rev, 2.5)),
                float(np.percentile(boot_aucs_rev, 97.5)),
            ) if boot_aucs_rev else (np.nan, np.nan),
        },
    }


def internal_validation(X, y, cv_folds=5, classifier="logistic", seed=42):
    """Stratified k-fold CV within a single cohort.

    Returns mean AUC and per-fold AUCs.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_aucs = []

    for train_idx, test_idx in skf.split(X, y):
        prob = _fit_predict(X[train_idx], y[train_idx], X[test_idx], classifier)
        fold_aucs.append(roc_auc_score(y[test_idx], prob))

    return {
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "fold_aucs": fold_aucs,
    }
