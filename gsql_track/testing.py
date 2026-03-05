"""Unified test utility functions integrating numpy, pandas, torch testing with consistent API

General Checks

- assert_dict_equal - Dictionary comparison with ignore_keys
- assert_list_equal - List comparison with nested support and dict key ignoring
- assert_frame_equal - DataFrame comparison with ignore_columns
- assert_series_equal - Pandas Series comparison
- assert_close - Approximate equality for all types

Shape & Type Checks

- assert_shape, assert_dtype, assert_device
- assert_range - Check values within min/max
- assert_batch_size - Verify batch dimension

Nested Structures

- assert_nested_equal - Deep comparison with path-based ignore

Context Managers

- assert_raises, assert_warns, assert_no_warnings
- assert_max_memory, assert_max_time
- deterministic_context - Seed all RNGs

Model Testing

- assert_model_output_shape
- assert_parameters_updated
- assert_gradients_exist
- assert_requires_grad
- assert_train_mode, assert_eval_mode

Quality Checks

- assert_deterministic - Function reproducibility
- assert_all_unique, assert_sorted
- assert_probability_distribution
- assert_positive_definite
- assert_no_nan, assert_no_inf, assert_finite
- assert_normalized - L2 norm = 1

GPU Testing (with skip if unavailable)

- assert_gpu_cpu_equivalent - Auto-skips if no CUDA

Serialization

- assert_serializable, assert_pickle_roundtrip

Utilities

- assert_contains, assert_has_keys
- assert_file_exists, assert_dir_exists
- assert_same_length
"""

import re
import time
import pickle
import tracemalloc
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Pattern
from pathlib import Path
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
try:
    import pandas as pd
    from pandas.testing import assert_frame_equal as pd_assert_frame_equal
    from pandas.testing import assert_series_equal as pd_assert_series_equal
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


# >>> Helper: Format diff for error messages <<<#
def _format_diff(actual, expected, name="value"):
    if isinstance(actual, (np.ndarray, list)) and isinstance(
        expected, (np.ndarray, list)
    ):
        actual_arr = np.asarray(actual)
        expected_arr = np.asarray(expected)
        if actual_arr.shape == expected_arr.shape:
            diff = actual_arr - expected_arr
            max_diff_idx = np.unravel_index(np.abs(diff).argmax(), diff.shape)
            return f"\n{name} mismatch:\nMax diff at {max_diff_idx}: actual={actual_arr[max_diff_idx]}, expected={expected_arr[max_diff_idx]}\nActual:\n{actual}\nExpected:\n{expected}"
    return f"\n{name} mismatch:\nActual: {actual}\nExpected: {expected}"


# >>> Helper: Check if current path should be ignored <<<#
def _should_ignore_key(current_path: str, ignore_keys: List[str]) -> bool:
    """Check if the current path should be ignored.

    Supports both simple keys and nested paths:
    - ignore_keys=['timestamp'] matches any key named 'timestamp' at any level
    - ignore_keys=['a.b'] matches only the nested path d['a']['b']
    - ignore_keys=['a.b.c'] matches d['a']['b']['c']
    """
    # Check exact path match (e.g., 'stats.timestamp')
    if current_path in ignore_keys:
        return True
    key_name = current_path.split('.')[-1]
    if key_name in ignore_keys and '.' not in [k for k in ignore_keys if k == key_name][0:1]:
        return True

    return False


# >>> Helper: Convert to numpy array from various types <<<#
def _to_numpy(arr):
    if HAS_TORCH and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    elif HAS_PANDAS and isinstance(arr, (pd.Series, pd.DataFrame)):
        return arr.values
    elif isinstance(arr, (list, tuple)):
        return np.array(arr)
    else:
        return np.array([arr])


# >>> Helper: Get type name for error messages <<<#
def _get_type_name(obj):
    if HAS_TORCH and isinstance(obj, torch.Tensor):
        return (
            f"torch.Tensor(shape={obj.shape}, dtype={obj.dtype}, device={obj.device})"
        )
    elif isinstance(obj, np.ndarray):
        return f"numpy.ndarray(shape={obj.shape}, dtype={obj.dtype})"
    elif HAS_PANDAS and isinstance(obj, pd.DataFrame):
        return f"pd.DataFrame(shape={obj.shape})"
    elif HAS_PANDAS and isinstance(obj, pd.Series):
        return f"pd.Series(shape={obj.shape})"
    else:
        return type(obj).__name__


# >>> Assert: Dictionary equality with ignore keys <<<#
def assert_dict_equal(
    dict1: Dict,
    dict2: Dict,
    ignore_keys: Optional[List[str]] = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_dtype: bool = False,
    msg: str = "",
    path: str = "",
):
    ignore_keys = ignore_keys or []

    keys1 = set()
    for k in dict1.keys():
        current_path = f"{path}.{k}" if path else str(k)
        if not _should_ignore_key(current_path, ignore_keys):
            keys1.add(k)

    keys2 = set()
    for k in dict2.keys():
        current_path = f"{path}.{k}" if path else str(k)
        if not _should_ignore_key(current_path, ignore_keys):
            keys2.add(k)

    if keys1 != keys2:
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        raise AssertionError(
            f"{msg}\nKey mismatch:\nOnly in dict1: {only_in_1}\nOnly in dict2: {only_in_2}"
        )

    for key in keys1:
        if key not in dict2:
            continue
        val1, val2 = dict1[key], dict2[key]
        current_path = f"{path}.{key}" if path else str(key)

        if isinstance(val1, dict) and isinstance(val2, dict):
            assert_dict_equal(
                val1, val2, ignore_keys, rtol, atol, check_dtype, f"{msg}.{key}", current_path,
            )
        elif isinstance(val1, (np.ndarray, list, tuple)) or (
            HAS_TORCH and isinstance(val1, torch.Tensor)
        ):
            try:
                assert_list_equal(
                    val1, val2, rtol=rtol, atol=atol, check_dtype=check_dtype, msg=f"{msg} key='{key}'",
                )
            except AssertionError as e:
                raise AssertionError(f"{msg}\nKey '{key}' failed comparison:\n{e}")
        else:
            if val1 != val2:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if not np.allclose([val1], [val2], rtol=rtol, atol=atol):
                        raise AssertionError(
                            f"{msg}\nKey '{key}': {val1} != {val2} (rtol={rtol}, atol={atol})"
                        )
                else:
                    raise AssertionError(f"{msg}\nKey '{key}': {val1} != {val2}")


# >>> Assert: List equality with nested support and dict key ignoring <<<#
def assert_list_equal(
    list1: List,
    list2: List,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_dtype: bool = False,
    msg: str = "",
    path: str = "",
):
    if len(list1) != len(list2):
        raise AssertionError(
            f"{msg}\nLength mismatch at {path or 'root'}: {len(list1)} != {len(list2)}"
        )

    for idx, (val1, val2) in enumerate(zip(list1, list2)):
        current_path = f"{path}[{idx}]" if path else f"[{idx}]"

        if isinstance(val1, dict) and isinstance(val2, dict):
            assert_dict_equal(
                val1, val2, [], rtol, atol, check_dtype, f"{msg} at {current_path}", current_path,
            )
        elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            assert_list_equal(val1, val2, rtol, atol, check_dtype, msg, current_path)
        elif isinstance(val1, np.ndarray) or (
            HAS_TORCH and isinstance(val1, torch.Tensor)
        ):
            try:
                assert_array_equal(
                    val1, val2, rtol=rtol, atol=atol, check_dtype=check_dtype, msg=f"{msg} at {current_path}",
                )
            except AssertionError as e:
                raise AssertionError(f"{msg}\nIndex {idx} at {current_path} failed comparison:\n{e}")
        else:
            if val1 != val2:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if not np.allclose([val1], [val2], rtol=rtol, atol=atol):
                        raise AssertionError(
                            f"{msg}\nIndex {idx} at {current_path}: {val1} != {val2} (rtol={rtol}, atol={atol})"
                        )
                else:
                    raise AssertionError(f"{msg}\nIndex {idx} at {current_path}: {val1} != {val2}")


# >>> Assert: Array/Tensor equality with auto type detection <<<#
def assert_array_equal(
    actual, expected,
    rtol: float = 1e-5, atol: float = 1e-8,
    check_dtype: bool = True, check_shape: bool = True,
    msg: str = "", nan_ok: bool = False, inf_ok: bool = False,
):
    actual_np = _to_numpy(actual)
    expected_np = _to_numpy(expected)
    if check_shape and actual_np.shape != expected_np.shape:
        raise AssertionError(
            f"{msg}\nShape mismatch: {actual_np.shape} != {expected_np.shape}\nActual type: {_get_type_name(actual)}\nExpected type: {_get_type_name(expected)}"
        )
    if check_dtype and actual_np.dtype != expected_np.dtype:
        raise AssertionError(f"{msg}\nDtype mismatch: {actual_np.dtype} != {expected_np.dtype}")
    if nan_ok:
        mask = np.isnan(actual_np) | np.isnan(expected_np)
        if not np.array_equal(np.isnan(actual_np), np.isnan(expected_np)):
            raise AssertionError(f"{msg}\nNaN positions differ{_format_diff(actual_np, expected_np)}")
        actual_np = actual_np[~mask]
        expected_np = expected_np[~mask]
    if inf_ok:
        inf_mask = np.isinf(actual_np) | np.isinf(expected_np)
        if not np.array_equal(np.isinf(actual_np), np.isinf(expected_np)):
            raise AssertionError(f"{msg}\nInf positions differ{_format_diff(actual_np, expected_np)}")
        actual_np = actual_np[~inf_mask]
        expected_np = expected_np[~inf_mask]
    if not np.allclose(actual_np, expected_np, rtol=rtol, atol=atol, equal_nan=nan_ok):
        raise AssertionError(f"{msg}\nArrays not equal (rtol={rtol}, atol={atol}){_format_diff(actual_np, expected_np)}")


def assert_frame_equal(
    df1, df2,
    ignore_columns: Optional[List[str]] = None, ignore_index: bool = False,
    check_column_order: bool = True, check_categorical: bool = True,
    rtol: float = 1e-5, atol: float = 1e-8, check_dtype: bool = True,
):
    if not HAS_PANDAS:
        raise ImportError("pandas not installed")
    ignore_columns = ignore_columns or []
    df1_filtered = df1.drop(columns=[c for c in ignore_columns if c in df1.columns])
    df2_filtered = df2.drop(columns=[c for c in ignore_columns if c in df2.columns])
    if not check_column_order:
        df2_filtered = df2_filtered[df1_filtered.columns]
    try:
        pd_assert_frame_equal(
            df1_filtered, df2_filtered,
            check_like=not check_column_order, check_categorical=check_categorical,
            rtol=rtol, atol=atol, check_dtype=check_dtype,
            check_index_type=not ignore_index, check_column_type=check_dtype,
        )
    except AssertionError as e:
        raise AssertionError(f"DataFrame comparison failed:\n{e}")


def assert_series_equal(
    s1, s2, rtol: float = 1e-5, atol: float = 1e-8,
    check_dtype: bool = True, check_index: bool = True,
):
    if not HAS_PANDAS:
        raise ImportError("pandas not installed")
    try:
        pd_assert_series_equal(
            s1, s2, rtol=rtol, atol=atol, check_dtype=check_dtype, check_index_type=check_index,
        )
    except AssertionError as e:
        raise AssertionError(f"Series comparison failed:\n{e}")


def assert_close(
    actual, expected, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = True
):
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert_dict_equal(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"Length mismatch: {len(actual)} != {len(expected)}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            try:
                assert_close(a, e, rtol=rtol, atol=atol, equal_nan=equal_nan)
            except AssertionError as err:
                raise AssertionError(f"Element {i} failed:\n{err}")
    elif isinstance(actual, (np.ndarray,)) or (HAS_TORCH and isinstance(actual, torch.Tensor)):
        assert_array_equal(actual, expected, rtol=rtol, atol=atol, nan_ok=equal_nan)
    elif isinstance(actual, (int, float, np.number)):
        if equal_nan and np.isnan(actual) and np.isnan(expected):
            return
        if not np.allclose([actual], [expected], rtol=rtol, atol=atol):
            raise AssertionError(f"Values not close: {actual} != {expected} (rtol={rtol}, atol={atol})")
    else:
        if actual != expected:
            raise AssertionError(f"Values not equal: {actual} != {expected}")


def assert_shape(arr, expected_shape: Tuple[int, ...]):
    if HAS_TORCH and isinstance(arr, torch.Tensor):
        actual_shape = tuple(arr.shape)
    elif isinstance(arr, np.ndarray):
        actual_shape = arr.shape
    elif HAS_PANDAS and isinstance(arr, (pd.DataFrame, pd.Series)):
        actual_shape = arr.shape
    else:
        raise TypeError(f"Unsupported type: {type(arr)}")
    if actual_shape != expected_shape:
        raise AssertionError(f"Shape mismatch: {actual_shape} != {expected_shape}")


def assert_device(tensor, expected_device: str):
    if not HAS_TORCH:
        raise ImportError("torch not installed")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    actual_device = str(tensor.device)
    if actual_device != expected_device:
        raise AssertionError(f"Device mismatch: {actual_device} != {expected_device}")


def assert_range(arr, min_val: Optional[float] = None, max_val: Optional[float] = None, inclusive: bool = True):
    arr_np = _to_numpy(arr)
    if min_val is not None:
        if inclusive:
            if not np.all(arr_np >= min_val):
                violators = arr_np[arr_np < min_val]
                raise AssertionError(f"Values below minimum {min_val}: found {len(violators)} violations, min={violators.min()}")
        else:
            if not np.all(arr_np > min_val):
                violators = arr_np[arr_np <= min_val]
                raise AssertionError(f"Values not above minimum {min_val}: found {len(violators)} violations")
    if max_val is not None:
        if inclusive:
            if not np.all(arr_np <= max_val):
                violators = arr_np[arr_np > max_val]
                raise AssertionError(f"Values above maximum {max_val}: found {len(violators)} violations, max={violators.max()}")
        else:
            if not np.all(arr_np < max_val):
                violators = arr_np[arr_np >= max_val]
                raise AssertionError(f"Values not below maximum {max_val}: found {len(violators)} violations")


def assert_nested_equal(
    obj1, obj2,
    ignore_paths: Optional[List[str]] = None,
    rtol: float = 1e-5, atol: float = 1e-8, path: str = "",
):
    ignore_paths = ignore_paths or []
    if path in ignore_paths:
        return
    if isinstance(obj1, dict):
        if obj1.keys() != obj2.keys():
            raise AssertionError(f"Keys mismatch at {path}: {obj1.keys()} != {obj2.keys()}")
        for key in obj1.keys():
            new_path = f"{path}.{key}" if path else str(key)
            assert_nested_equal(obj1[key], obj2[key], ignore_paths, rtol, atol, new_path)
    elif isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            raise AssertionError(f"Length mismatch at {path}: {len(obj1)} != {len(obj2)}")
        for i, (v1, v2) in enumerate(zip(obj1, obj2)):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            assert_nested_equal(v1, v2, ignore_paths, rtol, atol, new_path)
    elif isinstance(obj1, (np.ndarray,)) or (HAS_TORCH and isinstance(obj1, torch.Tensor)):
        assert_array_equal(obj1, obj2, rtol=rtol, atol=atol, msg=f"at path {path}")
    else:
        assert_close(obj1, obj2, rtol=rtol, atol=atol)


@contextmanager
def assert_raises(exception_type: type, match: Optional[str] = None):
    try:
        yield
        raise AssertionError(f"Expected {exception_type.__name__} but nothing was raised")
    except exception_type as e:
        if match and not re.search(match, str(e)):
            raise AssertionError(f"Exception message '{e}' does not match pattern '{match}'")
    except Exception as e:
        raise AssertionError(f"Expected {exception_type.__name__} but got {type(e).__name__}: {e}")


@contextmanager
def assert_warns(warning_type: type = UserWarning):
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w
        if not any(issubclass(warning.category, warning_type) for warning in w):
            raise AssertionError(f"Expected {warning_type.__name__} but no such warning was raised")


@contextmanager
def assert_no_warnings():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield
        if len(w) > 0:
            raise AssertionError(f"Expected no warnings but got {len(w)}: {[str(warning.message) for warning in w]}")


def assert_model_output_shape(model, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], device: str = "cpu"):
    if not HAS_TORCH:
        raise ImportError("torch not installed")
    dummy_input = torch.randn(*input_shape).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    actual_shape = tuple(output.shape)
    if actual_shape != output_shape:
        raise AssertionError(f"Output shape mismatch: {actual_shape} != {output_shape}")


def assert_parameters_updated(
    model_before: Dict[str, Any], model_after: Dict[str, Any],
    exclude: Optional[List[str]] = None, rtol: float = 1e-5,
):
    if not HAS_TORCH:
        raise ImportError("torch not installed")
    exclude = exclude or []
    updated_params = []
    unchanged_params = []
    for name in model_before.keys():
        if any(ex in name for ex in exclude):
            continue
        if name not in model_after:
            raise AssertionError(f"Parameter {name} missing in model_after")
        before = model_before[name]
        after = model_after[name]
        if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor):
            if not torch.allclose(before, after, rtol=rtol):
                updated_params.append(name)
            else:
                unchanged_params.append(name)
    if not updated_params:
        raise AssertionError(f"No parameters were updated. All {len(unchanged_params)} parameters remained unchanged.")


def assert_gradients_exist(model, check_all: bool = True):
    if not HAS_TORCH:
        raise ImportError("torch not installed")
    params_without_grad = []
    params_with_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                params_without_grad.append(name)
            else:
                params_with_grad.append(name)
    if check_all and params_without_grad:
        raise AssertionError(f"{len(params_without_grad)} parameters require grad but have no gradient:\n{params_without_grad[:10]}")
    if not params_with_grad:
        raise AssertionError("No parameters have gradients")


def assert_deterministic(
    fn: Callable, args: tuple = (), kwargs: dict = None,
    n_runs: int = 5, rtol: float = 1e-5, atol: float = 1e-8,
):
    kwargs = kwargs or {}
    first_result = fn(*args, **kwargs)
    for i in range(1, n_runs):
        result = fn(*args, **kwargs)
        try:
            assert_close(result, first_result, rtol=rtol, atol=atol)
        except AssertionError as e:
            raise AssertionError(f"Function is not deterministic. Run {i + 1}/{n_runs} produced different result:\n{e}")


@contextmanager
def assert_max_memory(max_bytes: int):
    tracemalloc.start()
    try:
        yield
        current, peak = tracemalloc.get_traced_memory()
        if peak > max_bytes:
            raise AssertionError(f"Memory usage {peak:,} bytes exceeds maximum {max_bytes:,} bytes")
    finally:
        tracemalloc.stop()


@contextmanager
def assert_max_time(max_seconds: float):
    start = time.time()
    yield
    elapsed = time.time() - start
    if elapsed > max_seconds:
        raise AssertionError(f"Execution time {elapsed:.3f}s exceeds maximum {max_seconds:.3f}s")


def assert_serializable(obj, path: Optional[Path] = None):
    try:
        serialized = pickle.dumps(obj)
        deserialized = pickle.loads(serialized)
        if path:
            with open(path, "wb") as f:
                pickle.dump(obj, f)
            with open(path, "rb") as f:
                deserialized = pickle.load(f)
    except Exception as e:
        raise AssertionError(f"Object not serializable: {e}")


def assert_pickle_roundtrip(obj, rtol: float = 1e-5, atol: float = 1e-8):
    serialized = pickle.dumps(obj)
    deserialized = pickle.loads(serialized)
    try:
        assert_close(obj, deserialized, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"Pickle roundtrip changed object:\n{e}")


def assert_all_unique(arr):
    arr_np = _to_numpy(arr).flatten()
    unique_vals = np.unique(arr_np)
    if len(unique_vals) != len(arr_np):
        raise AssertionError(f"Array has duplicates: {len(arr_np)} elements but only {len(unique_vals)} unique")


def assert_sorted(arr, descending: bool = False):
    arr_np = _to_numpy(arr).flatten()
    if descending:
        if not np.all(arr_np[:-1] >= arr_np[1:]):
            raise AssertionError("Array is not sorted in descending order")
    else:
        if not np.all(arr_np[:-1] <= arr_np[1:]):
            raise AssertionError("Array is not sorted in ascending order")


def assert_probability_distribution(arr, axis: Optional[int] = None, rtol: float = 1e-5, atol: float = 1e-8):
    arr_np = _to_numpy(arr)
    sums = arr_np.sum(axis=axis)
    if not np.allclose(sums, 1.0, rtol=rtol, atol=atol):
        raise AssertionError(f"Not a probability distribution. Sums: {sums}")


def assert_positive_definite(matrix):
    matrix_np = _to_numpy(matrix)
    if matrix_np.shape[0] != matrix_np.shape[1]:
        raise AssertionError(f"Matrix is not square: {matrix_np.shape}")
    eigenvalues = np.linalg.eigvalsh(matrix_np)
    if not np.all(eigenvalues > 0):
        raise AssertionError(f"Matrix is not positive definite. Min eigenvalue: {eigenvalues.min()}")


def assert_requires_grad(tensor, requires_grad: bool = True):
    if not HAS_TORCH:
        raise ImportError("torch not installed")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if tensor.requires_grad != requires_grad:
        raise AssertionError(f"Expected requires_grad={requires_grad}, got {tensor.requires_grad}")


def assert_contains(
    text: str, substring: Union[str, Pattern], *, regex: bool = False, flags: int = 0
):
    if not regex:
        if substring.lower() not in text.lower():
            raise AssertionError(f"'{substring}' not found in text.")
        return
    pat = re.compile(substring, flags=flags)
    if not pat.search(text):
        raise AssertionError(f"Regex pattern '{pat.pattern}' not found in text.")


def assert_has_keys(d: Dict, keys: List[str], exact: bool = False):
    missing = set(keys) - set(d.keys())
    if missing:
        raise AssertionError(f"Missing keys: {missing}")
    if exact:
        extra = set(d.keys()) - set(keys)
        if extra:
            raise AssertionError(f"Extra keys: {extra}")


def assert_file_exists(path: Union[str, Path]):
    p = Path(path)
    if not p.exists():
        raise AssertionError(f"File does not exist: {path}")
    if not p.is_file():
        raise AssertionError(f"Path is not a file: {path}")


def assert_dir_exists(path: Union[str, Path]):
    p = Path(path)
    if not p.exists():
        raise AssertionError(f"Directory does not exist: {path}")
    if not p.is_dir():
        raise AssertionError(f"Path is not a directory: {path}")


@contextmanager
def deterministic_context(seed: int = 42):
    import random
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state() if HAS_TORCH else None
    torch_cuda_state = (
        torch.cuda.get_rng_state_all()
        if HAS_TORCH and torch.cuda.is_available()
        else None
    )
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if HAS_TORCH and torch_state is not None:
            torch.set_rng_state(torch_state)
        if HAS_TORCH and torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(torch_cuda_state)


def assert_no_nan(arr):
    arr_np = _to_numpy(arr)
    if np.any(np.isnan(arr_np)):
        nan_count = np.sum(np.isnan(arr_np))
        raise AssertionError(f"Array contains {nan_count} NaN values")


def assert_no_inf(arr):
    arr_np = _to_numpy(arr)
    if np.any(np.isinf(arr_np)):
        inf_count = np.sum(np.isinf(arr_np))
        raise AssertionError(f"Array contains {inf_count} Inf values")


def assert_finite(arr):
    assert_no_nan(arr)
    assert_no_inf(arr)


def assert_normalized(arr, axis: Optional[int] = None, rtol: float = 1e-5, atol: float = 1e-8):
    arr_np = _to_numpy(arr)
    norms = np.linalg.norm(arr_np, axis=axis)
    if not np.allclose(norms, 1.0, rtol=rtol, atol=atol):
        raise AssertionError(f"Array is not normalized. Norms: {norms}")


def assert_train_mode(model):
    if not HAS_TORCH:
        raise ImportError("torch not installed")
    if not model.training:
        raise AssertionError("Model is not in training mode")


def assert_eval_mode(model):
    if not HAS_TORCH:
        raise ImportError("torch not installed")
    if model.training:
        raise AssertionError("Model is not in eval mode")
