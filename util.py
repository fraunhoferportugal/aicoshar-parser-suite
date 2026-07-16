import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def set_seed(seed: int = 42) -> np.random.Generator:
    """Set random seed for numpy.

    https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
    """
    rng = np.random.default_rng(seed)
    return rng


def convert_ndarrays(obj):
    """Recursively convert NumPy arrays to lists to ensure JSON
    serializability.

    Args:
        obj: The input object potentially containing NumPy arrays.

    Returns:
        A JSON-serializable version of the object.

    Raises:
        TypeError: If an unsupported type is encountered.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_to_json(data, directory, logger, indent=2):
    """Save a Python dictionary to 'train_test.json' in the given directory.
    Ensures NumPy arrays are converted and raises clear exceptions.

    Args:
        data (dict): The data to serialize.
        directory (str or Path): The target directory for saving.
        logger: Logger instance (e.g., from loguru or logging).
        indent (int): Number of spaces for indentation in the JSON file.

    Raises:
        TypeError: If data contains unserializable types.
        ValueError: If serialization fails for other reasons.
        OSError: If file writing fails.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    filepath = directory / "train_test.json"

    try:
        clean_data = convert_ndarrays(data)
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=indent)

    except TypeError as e:
        logger.error(f"TypeError during JSON serialization: {e}")
        raise TypeError(f"Error serializing data to JSON: {e}") from e
    except ValueError as e:
        logger.error(f"ValueError during JSON serialization: {e}")
        raise ValueError(f"Invalid value during JSON serialization: {e}") from e
    except OSError as e:
        logger.error(f"I/O error writing to {filepath}: {e}")
        raise OSError(f"Failed to write file to {filepath}: {e}") from e


def random_train_test_split_ids(
    ids: List[str],
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Randomly splits a list of IDs into train and test sets.

    Parameters:
    - ids (List[str]): List of user or sample IDs.
    - test_ratio (float): Proportion of IDs to include in the test set.
    - seed (int): Random seed for reproducibility.

    Returns:
    - (train_ids, test_ids): Tuple of lists.
    """
    ids = ids.copy()
    random.seed(seed)
    random.shuffle(ids)

    split_idx = int(len(ids) * (1 - test_ratio))
    train_ids = ids[:split_idx]
    test_ids = ids[split_idx:]

    return train_ids, test_ids
