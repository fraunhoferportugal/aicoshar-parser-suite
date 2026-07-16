#!/usr/bin/env python3

from __future__ import annotations

import io
import logging
import mimetypes
import os
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import kagglehub
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

try:
    import rarfile  # type: ignore

    _RAR_AVAILABLE = True
except Exception:
    _RAR_AVAILABLE = False

# Project config
from config import data_raw_dir  # expected to be str or Path

# ------------------------- Configurable constants -------------------------

DATASETS: dict[str, str] = {
    "CHARM": [
        "https://zenodo.org/records/4642560/files/CHARM_v1.1_accelerometer.csv?download=1",
        "https://zenodo.org/records/4642560/files/CHARM_v1.1_gyroscope.csv?download=1",
        "https://zenodo.org/records/4642560/files/CHARM_dataset_v1.1_raw_about.txt?download=1",
    ],
    "DailySportsActivities": "obirgul/daily-and-sports-activities",
    "ExtraSensory": [
        "http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip",
        "http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip",
        "http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip",
        "http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.raw_magnet.zip",
    ],
    "FLAAP": "https://data.mendeley.com/public-api/zip/bdng756rgw/download/1",
    "HARSense": "nurulaminchoudhury/harsense-datatset",
    "HHAR": "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip",
    "HuGaDB": "https://github.com/romanchereshnev/HuGaDB/raw/refs/heads/master/HumanGaitDataBase.zip",
    "KuHAR": "https://data.mendeley.com/public-files/datasets/45f952y38r/files/d3126562-b795-4eba-8559-310a25859cc7/file_downloaded",
    "MHEALTH": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
    "MotionSense": [
        "https://github.com/mmalekzadeh/motion-sense/raw/refs/heads/master/data/A_DeviceMotion_data.zip",
        "https://github.com/mmalekzadeh/motion-sense/raw/refs/heads/master/data/B_Accelerometer_data.zip",
        "https://github.com/mmalekzadeh/motion-sense/raw/refs/heads/master/data/C_Gyroscope_data.zip",
        "https://github.com/mmalekzadeh/motion-sense/raw/refs/heads/master/data/data_subjects_info.csv",
    ],
    "OPPORTUNITY": "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip",
    "PAMAP2": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
    "RealWorld": "http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip",
    "Shoaib13": "https://www.utwente.nl/en/eemcs/ps/dataset-folder/activity-recognition-dataset-shoaib.rar",
    "Shoaib14": "https://www.utwente.nl/en/eemcs/ps/dataset-folder/sensors-activity-recognition-dataset-shoaib.rar",
    "Shoaib16": "https://www.utwente.nl/en/eemcs/ps/dataset-folder/ut-data-complex.rar",
    "UCIHAR": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "UniMiB-SHAR": "https://www.dropbox.com/2/sharing_receiving/generate_download_url",
    "USC-HAD": "https://sipi.usc.edu/had/USC-HAD.zip",
    "WISDM": "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
}

# Networking
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 60
TOTAL_RETRIES = 3
BACKOFF_FACTOR = 0.5

# Download
CHUNK_BYTES = 1 << 20  # 1 MiB

# Logging format
LOG_FORMAT = "[%(levelname)s] %(message)s"

# Extensions
ALLOWED_EXTENSIONS = {
    ".txt",
    ".csv",
    ".json",
    ".xml",
    ".tar",
    ".gz",
    ".zip",
    ".bz2",
    ".tbz2",
    ".rar",
    ".tgz",
    ".xz",
    ".txz",
}

# Exceptions
NON_DOWNLOADABLE_DATASETS = ["UniMiB-SHAR"]

# ------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="\n%(levelname)s: %(message)s",
        force=True,
    )


def remove_raw_data(name: str, base_dir: Path):

    if (base_dir / name).exists():
        shutil.rmtree(base_dir / name)
    else:
        logging.info(f"Directory {base_dir / name} does not exist, skipping raw data removal.")


def detect_exceptions(dataset):
    return dataset in NON_DOWNLOADABLE_DATASETS


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Conservative sanitization for filesystem names."""
    name = name.strip().replace(os.sep, "_")
    name = re.sub(r"[^A-Za-z0-9._\-+ ]+", "_", name)
    return name


def requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=TOTAL_RETRIES,
        read=TOTAL_RETRIES,
        connect=TOTAL_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(
        {
            "User-Agent": "aicoshar-dataset-downloader/1.0 (+https://example.org)",
            "Accept": "*/*",
        },
    )
    return s


def infer_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    fname = unquote(Path(path).name) or "download"
    return sanitize_filename(fname)


def infer_filename_from_response(resp: requests.Response) -> str:
    cd = resp.headers.get("Content-Disposition")
    if cd:
        match = re.search(r'filename="?(.+?)"?$', cd)
        if match:
            return match.group(1)
    return None


def infer_ext_from_response(url: str, resp: requests.Response) -> str:
    """Try to infer a sensible extension using (1) URL, (2) Content-Type, (3)
    Content-Disposition.

    Falls back to empty string if unknown.
    """
    # Content-Disposition
    cd = resp.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^\";]+)"?', cd)
    if match:
        cand = sanitize_filename(unquote(match.group(1)))
        ext = Path(cand).suffix
        if ext:
            return ext.lower()

    # URL path
    ext = Path(urlparse(url).path).suffix.lower()
    if ext:
        return ext

    # Content-Type
    ctype = resp.headers.get("Content-Type", "")
    if ctype:
        exts = mimetypes.guess_all_extensions(ctype)
        if exts:
            return exts[0].lower()

    return ""


def is_archive_ext(ext: str) -> bool:
    return ext in {".zip", ".rar", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"}


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve()
        target = target.resolve()
    except FileNotFoundError:
        # For some archives, target may not exist yet; fall back to lexical check
        directory = directory.absolute()
        target = (directory / target).absolute()
    return str(target).startswith(str(directory) + os.sep)


def _safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
    for member in tar.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f"Blocked path traversal attempt in tar member: {member.name}")
    tar.extractall(path=path)


def _safe_extract_zip(zf: zipfile.ZipFile, path: Path) -> None:
    for member in zf.infolist():
        member_path = path / member.filename
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f"Blocked path traversal attempt in zip member: {member.filename}")
    zf.extractall(path=path)


def _safe_extract_rar(rf: rarfile.RarFile, path: Path) -> None:
    for member in rf.infolist():
        member_path = path / member.filename
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f"Blocked path traversal attempt in rar member: {member.filename}")
    rf.extractall(path=str(path))


def is_url(url) -> bool:
    if not isinstance(url, str):
        return False

    parsed = urlparse(url)
    return parsed.scheme in ("http", "https")


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    ext = archive_path.suffix.lower()
    # Normalize common compressed tar extensions
    if archive_path.name.endswith((".tar.gz", ".tgz")):
        ext = ".tgz"
    elif archive_path.name.endswith((".tar.bz2", ".tbz2")):
        ext = ".tar.bz2"
    elif archive_path.name.endswith((".tar.xz", ".txz")):
        ext = ".tar.xz"
    elif not ext and zipfile.is_zipfile(archive_path):
        ext = ".zip"

    ensure_dir(out_dir)

    if ext == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            _safe_extract_zip(zf, out_dir)
    elif ext in {".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"}:
        mode = "r"
        if ext in {".tgz", ".tar.gz"}:
            mode = "r:gz"
        elif ext in {".tar.bz2", ".tbz2"}:
            mode = "r:bz2"
        elif ext in {".tar.xz", ".txz"}:
            mode = "r:xz"
        with tarfile.open(archive_path, mode) as tf:
            _safe_extract_tar(tf, out_dir)
    elif ext == ".rar":
        if not _RAR_AVAILABLE:
            raise RuntimeError(
                "RAR extraction requested but 'rarfile' is not installed or backend unrar/bsdtar is missing.",
            )
        with rarfile.RarFile(archive_path) as rf:
            _safe_extract_rar(rf, out_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")


def extract_all_archives(folder):
    for path in folder.iterdir():
        if path.is_file():
            ext = path.suffix.lower()
            if is_archive_ext(ext):
                out_dir = path.with_suffix("")
                extract_archive(path, out_dir)

                # recurse into the newly extracted folder
                extract_all_archives(out_dir)

        elif path.is_dir():
            # recurse into existing subfolders
            extract_all_archives(path)


def download_from_kaggle(handle: str, output_dir: str):
    return kagglehub.dataset_download(handle=handle, output_dir=output_dir)


def download_stream_to_file(
    session: requests.Session,
    url: str,
    dst_file: Path,
    *,
    desc: str | None = None,
) -> tuple[bool, str | None]:
    """Stream a URL to dst_file with a tqdm progress bar.

    Returns (ok, error_message).
    """
    tmp_file = dst_file.with_suffix(dst_file.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
            if r.status_code >= 400:
                return False, f"HTTP {r.status_code} for {url}"

            total = int(r.headers.get("Content-Length", "0")) or None
            if total is None:
                # Unknown length; tqdm will still show a spinner
                pass

            # Show per-file download progress
            progress_bar = tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc or "downloading",
                leave=False,
            )
            with open(tmp_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_BYTES):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            progress_bar.close()
        tmp_file.replace(dst_file)
        return True, None
    except requests.RequestException as e:
        return False, f"Request error for {url}: {e}"
    except Exception as e:
        return False, f"Unexpected error for {url}: {e}"
    finally:
        if tmp_file.exists():
            try:
                tmp_file.unlink(missing_ok=True)
            except Exception:
                pass


def download_and_extract_dataset(
    name: str,
    url: str,
    base_dir: Path,
) -> tuple[bool, str | None]:
    """
    Idempotently download and extract dataset:
      - If base_dir/name exists and is non-empty, skip.
      - Otherwise, download to a temp file and extract if it's an archive; if not an archive, just move into the folder.
    Returns (success, error_message).
    """

    if detect_exceptions(name):
        logging.warning(
            f"{name} dataset does not provide a direct download link. "
            f"Manual download is required through {DATASETS[name]}",
        )
        return

    session = requests_session()
    dataset_dir = base_dir / sanitize_filename(name)

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        logging.info(f"{name}: already present at {dataset_dir} — skipping download.")
        return True, None

    ensure_dir(dataset_dir)

    urls = [url] if isinstance(url, str) else url

    for i, url in enumerate(urls):

        # Download from link
        # Ensure url format for download
        if is_url(url):
            try:
                head = session.head(url, allow_redirects=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            except requests.RequestException:
                head = None

            # Choose filename
            filename = infer_filename_from_url(url)

            with session.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
                resp_filename = infer_filename_from_response(r)
            filename = resp_filename if resp_filename else filename

            ext = ""
            if head is not None and head.ok:
                ext = infer_ext_from_response(url, head)
                # Prefer inferred extension when HEAD knows more
                if ext and not filename.endswith(ext):
                    filename = filename + ext

            # Final target file path (for the original download)
            target_file = dataset_dir / filename

            # Optional: adjust name per file
            file_name = f"{name}_download_{i}" if len(urls) > 1 else name

            # Download
            logging.info(f"Downloading {name} from {url}")
            ok, err = download_stream_to_file(session, url, target_file, desc=file_name)
            if not ok:
                logging.warning(f"{file_name}: download failed — {err}")
                # Clean up empty folder
                if dataset_dir.exists() and not any(dataset_dir.iterdir()):
                    try:
                        dataset_dir.rmdir()
                    except Exception:
                        pass

                return False, err

        # Download from kaggle
        else:
            target_file = dataset_dir

            if os.listdir(dataset_dir):
                target_file = os.path.join(dataset_dir, "kaggle")
                os.makedirs(target_file, exist_ok=True)

            logging.info(f"Downloading {name} from kaggle handle {url}")
            download_from_kaggle(url, target_file)

    # Extraction
    for item in dataset_dir.iterdir():
        target_file = dataset_dir / item
        ext = Path(target_file.name).suffix.lower()
        valid_suffixes = [s for s in target_file.suffixes if s in ALLOWED_EXTENSIONS]
        ext = "".join(valid_suffixes)

        if not ext and zipfile.is_zipfile(target_file):
            ext = ".zip"
            target_file.rename(target_file.with_suffix(ext))
            target_file = target_file.with_suffix(ext)

        if is_archive_ext(ext):
            try:
                logging.info(f"{name}: extracting archive...")
                extract_archive(target_file, target_file.with_suffix(""))
                extract_all_archives(target_file.with_suffix(""))

            except Exception as e:
                logging.warning(f"{name}: extraction failed — {e}")
                return False, f"extraction failed: {e}"
        else:
            # Not an archive: the file itself is the dataset (e.g., CSV)
            logging.info(f"{name}: non-archive file saved to {target_file}")

    # Final sanity: dataset folder should not be empty
    if not any(dataset_dir.iterdir()):
        msg = "dataset folder is empty after download/extract"
        logging.warning(f"{name}: {msg}")
        return False, msg

    logging.info(f"{name}: ready at {dataset_dir}")
    return True, None


def main() -> int:
    setup_logging()

    base_dir = Path(data_raw_dir) if not isinstance(data_raw_dir, Path) else data_raw_dir
    ensure_dir(base_dir)

    logging.info(f"Raw data directory: {base_dir}")

    session = requests_session()

    results = {}
    for name, url in tqdm(DATASETS.items(), desc="Datasets", unit="ds"):
        logging.info(f"Downloading {name} from {url}")
        ok, err = download_and_extract_dataset(session, name, url, base_dir)
        results[name] = (ok, err)

    # Final report
    missing = [n for n, (ok, _) in results.items() if not ok]
    if missing:
        logging.warning("Some datasets failed or are missing:")
        for n in missing:
            _, err = results[n]
            logging.warning(f"  - {n}: {err or 'unknown error'}")
        logging.info("Done with warnings.")
        return 1
    else:
        logging.info("All datasets downloaded and prepared successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
