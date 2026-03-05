"""
Standardized metrics registry system for ML evaluation.

Provides torchmetrics-inspired system for registering and managing evaluation
metrics with clear contracts about input/output formats.
"""
import hashlib
import numpy as np
from contextlib import contextmanager
from enum import Enum
from inspect import Parameter, Signature
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Callable, Optional, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


#>>> Metric type definitions and metadata <<<
class MetricInputType(Enum):
    """Defines expected input format for metrics.

    Example
    -------
    >>> MetricInputType.LABELS  # 1D array: [0, 1, 2, 1]
    >>> MetricInputType.PROBABILITIES  # 2D array: [[0.8, 0.2], [0.3, 0.7]]
    """
    LABELS = "labels"
    PROBABILITIES = "probabilities"
    FLEXIBLE = "flexible"


@dataclass
class MetricMetadata:
    """Metadata describing a metric's behavior and requirements.

    Example
    -------
    >>> meta = MetricMetadata(name='acc', input_type=MetricInputType.LABELS, output_keys=['acc'])
    """
    name: str
    input_type: MetricInputType
    output_keys: list[str]
    task_type: str = "classification"
    description: str = ""

    def __repr__(self):
        return f"Metric({self.name}, input={self.input_type.value}, outputs={self.output_keys})"


#>>> Metric wrapper with input standardization <<<
class MetricWrapper:
    """Wrapper that standardizes metric inputs and provides metadata.

    Example
    -------
    >>> wrapper = MetricWrapper(func, metadata)
    >>> results = wrapper(labels=y_true, predictions=y_pred)
    """

    def __init__(self, func: Callable, metadata: MetricMetadata):
        self.func = func
        self.metadata = metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__signature__ = Signature([
            Parameter("labels", Parameter.KEYWORD_ONLY, annotation=np.ndarray),
            Parameter("predictions", Parameter.KEYWORD_ONLY, annotation=np.ndarray),
            Parameter("kwargs", Parameter.VAR_KEYWORD),
        ])

    def __call__(self, *, labels: np.ndarray, predictions: np.ndarray, **kwargs) -> Dict[str, float]:
        """Standardized interface: metric(labels, predictions)."""
        predictions = self._standardize_input(labels, predictions)
        result = self.func(labels, predictions, **kwargs)

        if not isinstance(result, dict):
            raise ValueError(f"Metric {self.metadata.name} must return a dict, got {type(result)}")

        return result

    def _standardize_input(self, labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Transform predictions to match metric's expected input type."""
        if self.metadata.input_type == MetricInputType.FLEXIBLE:
            return predictions
        elif self.metadata.input_type == MetricInputType.LABELS:
            if predictions.ndim == 2:
                predictions = predictions.argmax(axis=1)
            elif predictions.ndim != 1:
                raise ValueError(f"Metric {self.metadata.name} expects 1D labels, got shape {predictions.shape}")
            return predictions
        elif self.metadata.input_type == MetricInputType.PROBABILITIES:
            if predictions.ndim == 1:
                n_classes = len(np.unique(labels))
                predictions = np.eye(n_classes)[predictions]
            elif predictions.ndim != 2:
                raise ValueError(f"Metric {self.metadata.name} expects 2D probabilities, got shape {predictions.shape}")
            return predictions
        return predictions

    def __repr__(self):
        return f"<MetricWrapper: {self.metadata}>"


#>>> Global metric registry <<<
METRIC_REGISTRY: Dict[str, MetricWrapper] = {}
METRIC_METADATA: Dict[str, MetricMetadata] = {}


def register_metric(
    name: str,
    input_type: Union[MetricInputType, str] = MetricInputType.FLEXIBLE,
    output_keys: Optional[list[str]] = None,
    task_type: str = "classification",
    description: str = ""
):
    """Decorator to register metric computation functions with metadata.

    Example
    -------
    >>> @register_metric('acc', input_type=MetricInputType.PROBABILITIES, output_keys=['acc'])
    ... def accuracy(labels, predictions):
    ...     preds = predictions.argmax(axis=1)
    ...     return {'acc': accuracy_score(labels, preds)}
    """
    def decorator(func: Callable):
        if isinstance(input_type, str):
            input_type_enum = MetricInputType[input_type.upper()]
        else:
            input_type_enum = input_type

        metadata = MetricMetadata(
            name=name,
            input_type=input_type_enum,
            output_keys=output_keys or [],
            task_type=task_type,
            description=description or func.__doc__ or ""
        )

        wrapped = MetricWrapper(func, metadata)
        METRIC_REGISTRY[name] = wrapped
        METRIC_METADATA[name] = metadata

        return wrapped
    return decorator


def wrap_external_metric(
    name: str,
    func: Callable,
    input_type: Union[MetricInputType, str],
    output_keys: list[str],
    task_type: str = "classification",
    description: str = ""
) -> MetricWrapper:
    """Wrap an external metric function with standardization.

    Example
    -------
    >>> from wrench.evaluation import METRIC
    >>> wrapped = wrap_external_metric(
    ...     'acc', METRIC['acc'],
    ...     input_type=MetricInputType.PROBABILITIES,
    ...     output_keys=['acc']
    ... )
    >>> METRIC_REGISTRY['acc'] = wrapped
    """
    if isinstance(input_type, str):
        input_type_enum = MetricInputType[input_type.upper()]
    else:
        input_type_enum = input_type

    metadata = MetricMetadata(
        name=name,
        input_type=input_type_enum,
        output_keys=output_keys,
        task_type=task_type,
        description=description or func.__doc__ or ""
    )

    wrapped = MetricWrapper(func, metadata)
    METRIC_REGISTRY[name] = wrapped
    METRIC_METADATA[name] = metadata

    return wrapped


def list_metrics(task_type: Optional[str] = None, input_type: Optional[MetricInputType] = None) -> Dict[str, MetricMetadata]:
    """List all registered metrics, optionally filtered by task_type or input_type.

    Example
    -------
    >>> metrics = list_metrics(task_type='classification')
    >>> metrics = list_metrics(input_type=MetricInputType.LABELS)
    """
    filtered = {}
    for name, metadata in METRIC_METADATA.items():
        if task_type and metadata.task_type != task_type:
            continue
        if input_type and metadata.input_type != input_type:
            continue
        filtered[name] = metadata
    return filtered


#>>> Built-in metrics <<<
@register_metric(
    name='simple_classify',
    input_type=MetricInputType.FLEXIBLE,
    output_keys=['acc', 'prec', 'recall', 'f1'],
    task_type='classification',
    description='Compute basic classification metrics (acc, precision, recall, f1)'
)
def compute_basic_classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, average: str = "weighted"
) -> Dict[str, float]:
    """Compute basic classification metrics."""
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    return {
        'acc': float(accuracy),
        'prec': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


@register_metric(
    name='simple_regress',
    input_type=MetricInputType.FLEXIBLE,
    output_keys=['mse', 'mae', 'r2'],
    task_type='regression',
    description='Compute basic regression metrics (MSE, MAE, R²)'
)
def compute_regression_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    """Compute basic regression metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=1)
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2)
    }


def get_metric(name: str) -> MetricWrapper:
    """Get metric wrapper by name.

    Example
    -------
    >>> metric_fn = get_metric('acc')
    >>> result = metric_fn(labels=y_true, predictions=y_pred)
    """
    if name not in METRIC_REGISTRY:
        available = ', '.join(METRIC_REGISTRY.keys())
        raise KeyError(f"Metric '{name}' not found. Available: {available}")
    return METRIC_REGISTRY[name]


#>>> Embedding cache context manager <<<
@contextmanager
def embedding_cache(model, texts: list[str], model_name: str, cache_dir: str = "/tmp/emb_cache", force: bool = False):
    """Context manager yielding a cached proxy for a sentence transformer model.

    Cache key: md5(model_name + joined texts) → {cache_dir}/emb_{key}.npy
    On cache hit the proxy's .encode() returns the cached array/tensor directly.
    On cache miss it calls through to the real model and saves the result.

    Example
    -------
    >>> with embedding_cache(self.st_model, texts, 'all-MiniLM-L6-v2') as model:
    ...     embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    """
    key      = hashlib.md5((model_name + "\n".join(texts)).encode()).hexdigest()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    path = cache_path / f"emb_{key}.npy"

    class _CachedProxy:
        def encode(self, texts, **kwargs):
            if not force and path.exists():
                arr = np.load(str(path))
                if kwargs.get("convert_to_tensor"):
                    import torch
                    return torch.from_numpy(arr)
                return arr
            result = model.encode(texts, **kwargs)
            try:
                import torch
                arr = result.cpu().numpy() if isinstance(result, torch.Tensor) else np.asarray(result)
            except ImportError:
                arr = np.asarray(result)
            np.save(str(path), arr)
            return result

    yield _CachedProxy()


#>>> Valid-test correlation via frozen embeddings + LR sweep <<<
def val_test_corr(
    X_train, y_train,
    X_valid, y_valid,
    X_test,  y_test,
    model_name: str = "all-MiniLM-L6-v2",
    C_values: Optional[list] = None,
    seeds: Optional[list] = None,
    score: str = "auto",
    cache_dir: str = "/tmp/emb_cache",
    force: bool = False,
    dimensions: Optional[int] = None,
) -> dict:
    """Check whether validation performance predicts test performance.
    @param score : 'acc', 'f1', or 'auto'
        'auto' picks weighted-F1 when max_class / min_class > 2, else acc.
    @param dimensions : For OpenAI models, request reduced dimensionality (e.g. 512).

    Example
    -------
    >>> result = val_test_corr(train_texts, y_train, valid_texts, y_valid, test_texts, y_test)
    >>> print(f"Spearman r={result['spearman_r']:.3f}  (score={result['score']})")
    >>> # OpenAI embedding with 512 dimensions
    >>> result = val_test_corr(..., model_name="text-embedding-3-small", dimensions=512)
    """
    from collections import Counter
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression

    C_values = C_values or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    seeds    = seeds    or [0, 1, 2, 3, 4]

    if score == "auto":
        counts = Counter(y_train)
        ratio  = max(counts.values()) / max(min(counts.values()), 1)
        score  = "f1" if ratio > 2 else "acc"

    def _score(clf, X, y):
        preds = clf.predict(X)
        if score == "f1":
            from sklearn.metrics import f1_score
            return f1_score(y, preds, average="weighted", zero_division=0)
        return float(accuracy_score(y, preds))

    def _is_openai_model(name: str) -> bool:
        return name.startswith("text-embedding-")

    def _embed_openai(texts: list[str], model: str, dims: Optional[int]) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI()
        batch_size = 2048
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            kwargs = {"input": batch, "model": model}
            if dims is not None:
                kwargs["dimensions"] = dims
            resp = client.embeddings.create(**kwargs)
            all_embs.extend([e.embedding for e in resp.data])
        return np.array(all_embs)

    def _embed_splits(*splits):
        if not isinstance(splits[0][0], str):
            return [np.asarray(X) for X in splits]

        if _is_openai_model(model_name):
            cache_suffix = f"{model_name}_d{dimensions}" if dimensions else model_name
            results = []
            for X in splits:
                texts = list(X)
                key = hashlib.md5((cache_suffix + "\n".join(texts)).encode()).hexdigest()
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                path = cache_path / f"emb_{key}.npy"
                if not force and path.exists():
                    results.append(np.load(str(path)))
                else:
                    emb = _embed_openai(texts, model_name, dimensions)
                    np.save(str(path), emb)
                    results.append(emb)
            return results

        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(model_name)
        results = []
        for X in splits:
            texts = list(X)
            with embedding_cache(st_model, texts, model_name, cache_dir=cache_dir, force=force) as model:
                emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            try:
                import torch
                results.append(emb.cpu().numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb))
            except ImportError:
                results.append(np.asarray(emb))
        return results

    X_train, X_valid, X_test = _embed_splits(X_train, X_valid, X_test)
    y_train = np.asarray(y_train)
    y_valid = np.asarray(y_valid)
    y_test  = np.asarray(y_test)

    pairs = []
    n_train = len(X_train)
    for C in C_values:
        for seed in seeds:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n_train, size=n_train, replace=True)
            X_boot, y_boot = X_train[idx], y_train[idx]
            clf = LogisticRegression(C=C, random_state=seed, max_iter=1000)
            clf.fit(X_boot, y_boot)
            pairs.append({
                "C": C, "seed": seed,
                "valid": _score(clf, X_valid, y_valid),
                "test":  _score(clf, X_test,  y_test),
            })

    valid_scores = [p["valid"] for p in pairs]
    test_scores  = [p["test"]  for p in pairs]
    r, p_value   = spearmanr(valid_scores, test_scores)

    return {
        "pairs":      pairs,
        "spearman_r": float(r),
        "spearman_p": float(p_value),
        "score":      score,
        "n_configs":  len(pairs),
    }


__all__ = [
    "MetricInputType",
    "MetricMetadata",
    "MetricWrapper",
    "METRIC_REGISTRY",
    "METRIC_METADATA",
    "register_metric",
    "wrap_external_metric",
    "list_metrics",
    "get_metric",
    "embedding_cache",
    "val_test_corr",
]
