"""
Preprocess raw PROSTATEx MRI studies into lesion-centered patch tensors.

The script expects the public PROSTATEx training download to be unpacked locally,
including the metadata CSV files and the per-patient DICOM folders. It extracts
modality-specific 2D slices around each lesion location and writes stacked NumPy
patches plus a manifest that can be consumed by `examples.image_jepa.main`.
"""

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import pydicom
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

from eb_jepa.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODALITIES = ("t2", "adc", "dwi")
IMAGE_PATTERNS = ("*Images*.csv", "*images*.csv")
FINDING_PATTERNS = ("*Findings*.csv", "*findings*.csv")


def _parse_list_like(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _find_first(root_dir, patterns):
    root_dir = Path(root_dir)
    for pattern in patterns:
        matches = sorted(root_dir.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find files matching {patterns} under {root_dir}")


def _parse_ijk(value):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    if len(numbers) < 3:
        raise ValueError(f"Could not parse ijk triplet from: {value}")
    return tuple(int(round(float(num))) for num in numbers[:3])


def _parse_label(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    if text in {"0", "false", "no"}:
        return 0
    try:
        return int(round(float(text)))
    except ValueError as exc:
        raise ValueError(f"Unsupported ClinSig label: {value}") from exc


def _normalize_text(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _detect_modality(name):
    text = _normalize_text(name)
    if "adc" in text:
        return "adc"
    if "ktrans" in text:
        return "ktrans"
    if "t2" in text or "tse" in text:
        return "t2"
    if (
        "b800" in text
        or "b1000" in text
        or "bvalue" in text
        or "diff" in text
        or "dwi" in text
    ) and "adc" not in text:
        return "dwi"
    return None


def _load_csv_rows(path):
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def _score_series_row(row, modality):
    text = " ".join(
        [
            row.get("Name", ""),
            row.get("DCMSerDescr", ""),
            row.get("DCMSerUID", ""),
        ]
    ).lower()
    score = 0
    if modality == "t2":
        if "t2" in text:
            score += 5
        if "tra" in text or "ax" in text:
            score += 2
    elif modality == "adc":
        if "adc" in text:
            score += 5
    elif modality == "dwi":
        if "b800" in text or "b1000" in text:
            score += 5
        if "diff" in text or "dwi" in text:
            score += 3
        if "adc" in text:
            score -= 10
    return score


def _index_patient_series(patient_dir):
    patient_dir = Path(patient_dir)
    series_entries = []
    for root, _dirs, files in os.walk(patient_dir):
        if not files:
            continue
        root_path = Path(root)
        first_file = root_path / files[0]
        try:
            header = pydicom.dcmread(
                str(first_file),
                stop_before_pixels=True,
                force=True,
                specific_tags=["SeriesDescription", "SeriesNumber", "SeriesInstanceUID"],
            )
        except Exception:
            continue

        if not getattr(header, "SeriesInstanceUID", None):
            continue

        series_entries.append(
            {
                "dir": root_path,
                "series_number": str(getattr(header, "SeriesNumber", "")),
                "series_description": str(getattr(header, "SeriesDescription", "")),
                "series_uid": str(getattr(header, "SeriesInstanceUID", "")),
            }
        )

    return series_entries


def _match_series(series_rows, patient_series, modality):
    ranked_rows = sorted(
        series_rows,
        key=lambda row: _score_series_row(row, modality),
        reverse=True,
    )
    for row in ranked_rows:
        row_number = str(row.get("DCMSerNum", "")).strip()
        row_uid = str(row.get("DCMSerUID", "")).strip()
        row_name = row.get("Name", "")
        row_descr = row.get("DCMSerDescr", "")
        normalized_names = {_normalize_text(row_name), _normalize_text(row_descr)}

        for entry in patient_series:
            if row_uid and row_uid == entry["series_uid"]:
                return row, entry
            if row_number and row_number == entry["series_number"]:
                return row, entry
            entry_text = {
                _normalize_text(entry["series_description"]),
                _normalize_text(entry["series_uid"]),
            }
            if normalized_names & entry_text:
                return row, entry

    return None, None


def _load_series_array(series_dir):
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_dir))
    if not file_names:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 4:
        array = array[-1]
    if array.ndim != 3:
        raise ValueError(f"Expected 3D DICOM series in {series_dir}, got shape {array.shape}")
    return array


def _normalize_slice(slice_2d):
    slice_2d = slice_2d.astype(np.float32)
    lo, hi = np.percentile(slice_2d, [1, 99])
    if hi <= lo:
        if np.max(slice_2d) > np.min(slice_2d):
            lo = np.min(slice_2d)
            hi = np.max(slice_2d)
        else:
            return np.zeros_like(slice_2d, dtype=np.float32)
    slice_2d = np.clip((slice_2d - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return slice_2d


def _extract_patch(slice_2d, center_x, center_y, patch_size):
    half = patch_size // 2
    padded = np.pad(slice_2d, ((half, half), (half, half)), mode="edge")
    center_y += half
    center_x += half
    patch = padded[center_y - half : center_y + half, center_x - half : center_x + half]
    if patch.shape != (patch_size, patch_size):
        raise ValueError(f"Unexpected patch shape {patch.shape}")
    return patch


def _collect_finding_rows(image_rows):
    grouped = defaultdict(list)
    for row in image_rows:
        key = (row.get("ProxID", "").strip(), row.get("fid", "").strip())
        grouped[key].append(row)
    return grouped


def _build_split_map(patient_labels, val_ratio, seed):
    patients = sorted(patient_labels)
    if len(patients) < 2:
        return {patient: "train" for patient in patients}

    stratify = [patient_labels[patient] for patient in patients]
    try:
        train_patients, val_patients = train_test_split(
            patients,
            test_size=val_ratio,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_patients, val_patients = train_test_split(
            patients,
            test_size=val_ratio,
            random_state=seed,
        )

    split_map = {patient: "train" for patient in train_patients}
    split_map.update({patient: "val" for patient in val_patients})
    return split_map


def _compute_dataset_stats(manifest_rows, output_dir):
    train_arrays = []
    for row in manifest_rows:
        if row["split"] != "train":
            continue
        array = np.load(Path(output_dir) / row["image_path"])
        train_arrays.append(array)

    stacked = np.stack(train_arrays, axis=0)
    mean = stacked.mean(axis=(0, 2, 3)).tolist()
    std = stacked.std(axis=(0, 2, 3)).tolist()
    return {"mean": mean, "std": std}


def run(
    raw_root,
    output_dir,
    images_csv=None,
    findings_csv=None,
    patch_size=96,
    val_ratio=0.2,
    seed=42,
    modalities="t2,adc,dwi",
):
    """
    Preprocess the public PROSTATEx training set into patch tensors and a manifest.

    Args:
        raw_root: Root directory containing the unpacked PROSTATEx training data.
        output_dir: Where to save NumPy patches and the generated manifest.
        images_csv: Optional path to ProstateX-Images CSV.
        findings_csv: Optional path to ProstateX-Findings CSV with ClinSig labels.
        patch_size: Square lesion patch size, in source pixels, before training resize.
        val_ratio: Fraction of patients reserved for validation.
        seed: Random seed for the patient-level split.
        modalities: Comma-separated modality order, defaults to "t2,adc,dwi".
    """
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    images_csv = Path(images_csv) if images_csv else _find_first(raw_root, IMAGE_PATTERNS)
    findings_csv = (
        Path(findings_csv) if findings_csv else _find_first(raw_root, FINDING_PATTERNS)
    )
    modalities = tuple(_parse_list_like(modalities)) or DEFAULT_MODALITIES

    logger.info(f"Using images CSV: {images_csv}")
    logger.info(f"Using findings CSV: {findings_csv}")
    logger.info(f"Using modalities: {modalities}")

    image_rows = _load_csv_rows(images_csv)
    finding_rows = _load_csv_rows(findings_csv)
    finding_image_rows = _collect_finding_rows(image_rows)

    patient_dirs = {path.name: path for path in raw_root.rglob("ProstateX-*") if path.is_dir()}
    patient_series_index = {}

    patient_labels = {}
    records = []
    skipped = []

    for finding in finding_rows:
        patient_id = finding.get("ProxID", "").strip()
        finding_id = finding.get("fid", "").strip()
        label = _parse_label(finding.get("ClinSig", ""))

        if patient_id not in patient_dirs:
            skipped.append((patient_id, finding_id, "missing_patient_directory"))
            continue

        if patient_id not in patient_series_index:
            patient_series_index[patient_id] = _index_patient_series(patient_dirs[patient_id])

        series_rows = finding_image_rows.get((patient_id, finding_id), [])
        if not series_rows:
            skipped.append((patient_id, finding_id, "missing_image_rows"))
            continue

        modality_patches = []
        missing_modality = False
        for modality in modalities:
            modality_rows = [
                row
                for row in series_rows
                if _detect_modality(
                    " ".join([row.get("Name", ""), row.get("DCMSerDescr", "")])
                )
                == modality
            ]
            if not modality_rows:
                missing_modality = True
                skipped.append((patient_id, finding_id, f"missing_{modality}_metadata"))
                break

            matched_row, matched_series = _match_series(
                modality_rows, patient_series_index[patient_id], modality
            )
            if matched_row is None or matched_series is None:
                missing_modality = True
                skipped.append((patient_id, finding_id, f"missing_{modality}_dicom_series"))
                break

            try:
                col, row, slice_idx = _parse_ijk(matched_row.get("ijk", ""))
                volume = _load_series_array(matched_series["dir"])
                slice_idx = int(np.clip(slice_idx, 0, volume.shape[0] - 1))
                row = int(np.clip(row, 0, volume.shape[1] - 1))
                col = int(np.clip(col, 0, volume.shape[2] - 1))
                normalized_slice = _normalize_slice(volume[slice_idx])
                patch = _extract_patch(normalized_slice, center_x=col, center_y=row, patch_size=patch_size)
            except Exception as exc:
                missing_modality = True
                skipped.append((patient_id, finding_id, f"{modality}_extract_error:{exc}"))
                break

            modality_patches.append(patch)

        if missing_modality:
            continue

        stacked_patch = np.stack(modality_patches, axis=0).astype(np.float32)
        patch_rel_path = Path("patches") / f"{patient_id}_{finding_id}.npy"
        np.save(output_dir / patch_rel_path, stacked_patch)

        patient_labels[patient_id] = max(patient_labels.get(patient_id, 0), label)
        records.append(
            {
                "patient_id": patient_id,
                "finding_id": finding_id,
                "label": label,
                "image_path": str(patch_rel_path),
            }
        )

    if not records:
        raise RuntimeError("No PROSTATEx patches were generated. Check raw_root and CSV inputs.")

    split_map = _build_split_map(patient_labels, val_ratio=val_ratio, seed=seed)
    manifest_rows = []
    for record in records:
        manifest_rows.append(
            {
                **record,
                "split": split_map.get(record["patient_id"], "train"),
            }
        )

    manifest_path = output_dir / "prostatex_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        fieldnames = ["image_path", "label", "patient_id", "finding_id", "split"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    stats = _compute_dataset_stats(manifest_rows, output_dir)
    stats["modalities"] = list(modalities)
    stats["patch_size"] = patch_size
    stats["num_samples"] = len(manifest_rows)
    stats["num_train"] = sum(row["split"] == "train" for row in manifest_rows)
    stats["num_val"] = sum(row["split"] == "val" for row in manifest_rows)
    stats["skipped"] = len(skipped)

    with (output_dir / "dataset_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    with (output_dir / "skipped_records.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "finding_id", "reason"])
        writer.writerows(skipped)

    logger.info(f"Wrote manifest: {manifest_path}")
    logger.info(f"Saved dataset stats: {output_dir / 'dataset_stats.json'}")
    logger.info(
        "Training config hint: "
        f"data.manifest_path={manifest_path} "
        f"data.mean={stats['mean']} data.std={stats['std']}"
    )


if __name__ == "__main__":
    fire.Fire(run)
