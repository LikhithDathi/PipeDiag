from sklearn.datasets import make_classification

from pipediag.acquisition.split import SplitAcquirer


def load_data(context):
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    split_acquirer = SplitAcquirer(
        lineage=context.evidence_collector.lineage,
        registry=context.evidence_collector.registry,
        context_registry=context.dataset_contexts,
    )

    outputs, split_evidence = split_acquirer.split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    X_train, X_test, y_train, y_test = outputs
    return X_train, X_test, y_train, y_test, split_evidence
