import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List

import requests


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stream_download(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def download_nifti_samples(target_dir: Path) -> List[Path]:
    ensure_dir(target_dir)
    candidates = [
        (
            "https://github.com/fepegar/torchio-data/raw/main/colin27_t1_tal_lin.nii.gz",
            target_dir / "colin27_t1.nii.gz",
        ),
        (
            "https://raw.githubusercontent.com/ashwinsrnath/NIfTI-MRS-examples/main/svs_phantom/svs_phantom.nii.gz",
            target_dir / "phantom_svs.nii.gz",
        ),
       
        (
            "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.3/brain_mri.nii.gz",
            target_dir / "brain_mri.nii.gz",
        ),
    ]

    downloaded: List[Path] = []
    ok = False
    for url, dest in candidates:
        try:
            if not dest.exists():
                print(f"Downloading NIfTI: {url}")
                stream_download(url, dest)
            else:
                print(f"Already exists: {dest}")
            downloaded.append(dest)
            ok = True
            break
        except Exception as e:
            print(f"Failed NIfTI source {url}: {e}")

    if not ok:
        
        try:
            import nibabel as nib
            import numpy as np

            data = (np.random.rand(64, 64, 32) * 255).astype("uint8")
            img = nib.Nifti1Image(data, affine=np.eye(4))
            dest = target_dir / "synthetic_brain.nii.gz"
            nib.save(img, dest)
            print(f"Created synthetic NIfTI: {dest}")
            downloaded.append(dest)
        except Exception as e:
            print(f"Failed to create synthetic NIfTI: {e}")
            raise SystemExit("Unable to obtain any NIfTI sample.")
    return downloaded


def download_msd_spleen(target_dir: Path) -> List[Path]:
    """Download a real NIfTI dataset from Medical Segmentation Decathlon (Task09 Spleen)."""
    ensure_dir(target_dir)
    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    tar_path = target_dir / "Task09_Spleen.tar"
    root_extract = target_dir / "Task09_Spleen"
    if not root_extract.exists():
        try:
            print(f"Downloading MSD Spleen: {url}")
            stream_download(url, tar_path)
            print("Extracting MSD Spleen tar...")
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(target_dir)
            tar_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"Failed to download/extract MSD Spleen: {e}")
            return []
    else:
        print(f"MSD Spleen already extracted at {root_extract}")

    imagesTr = root_extract / "imagesTr"
    nifti_list = list(imagesTr.glob("*.nii.gz")) if imagesTr.exists() else []
    return nifti_list


def download_dicom_samples(target_dir: Path) -> List[Path]:
    ensure_dir(target_dir)
    
    cand_zips = [
        "https://github.com/dcm4che/dcm4chee-arc-light/files/11754355/dicomfiles.zip",
    ]
    single_dicoms = [
    
        "https://github.com/pydicom/pydicom-data/raw/main/data/CT_small.dcm",
        "https://github.com/pydicom/pydicom-data/raw/main/data/MR_small.dcm",
        "https://github.com/pydicom/pydicom-data/raw/main/data/SC_rgb.dcm",
    ]

    zip_path = target_dir / "dicom_samples.zip"
    extracted_dir = target_dir / "dicom_samples"

    moved: List[Path] = []
    if not extracted_dir.exists():
        ok = False
        for url in cand_zips:
            try:
                print(f"Downloading DICOM zip: {url}")
                stream_download(url, zip_path)
                print("Extracting DICOM zip...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extracted_dir)
                zip_path.unlink(missing_ok=True)
                ok = True
                break
            except Exception as e:
                print(f"Failed DICOM zip {url}: {e}")
        if not ok:
           
            for url in single_dicoms:
                try:
                    dest = target_dir / Path(url).name
                    print(f"Downloading single DICOM: {url}")
                    stream_download(url, dest)
                    moved.append(dest)
                    ok = True
                    break
                except Exception as e:
                    print(f"Failed single DICOM {url}: {e}")
            if not ok:
                raise SystemExit("Could not download any DICOM sample.")
    else:
        print(f"DICOM samples already extracted in {extracted_dir}")

    if extracted_dir.exists():
        for p in extracted_dir.rglob("*.dcm"):
            dest = target_dir / p.name
            if not dest.exists():
                shutil.copy2(p, dest)
                moved.append(dest)
        if not moved:
            moved = list(extracted_dir.rglob("*.dcm"))
    return moved


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    nifti_dir = data_dir / "nifti"
    dicom_dir = data_dir / "dicom"
    inputs_dir = data_dir / "inputs"
    ensure_dir(data_dir)
    ensure_dir(inputs_dir)

    print(f"Data root: {data_dir}")
    nifti_files = download_msd_spleen(nifti_dir)
    if not nifti_files:
        nifti_files = download_nifti_samples(nifti_dir)
    print(f"Downloaded/verified NIfTI files: {len(nifti_files)}")
    dicom_files = download_dicom_samples(dicom_dir)
    print(f"Downloaded/verified DICOM files: {len(dicom_files)}")

    try:
        if nifti_files:
            nifti_target = inputs_dir / nifti_files[0].name
            if not nifti_target.exists():
                shutil.copy2(nifti_files[0], nifti_target)
                print(f"Copied NIfTI to inputs: {nifti_target}")
            else:
                print(f"NIfTI already in inputs: {nifti_target}")
        if dicom_files:
            dicom_sample_dir = inputs_dir / "dicom_sample"
            ensure_dir(dicom_sample_dir)
       
            copied = 0
            for p in dicom_files:
                dest = dicom_sample_dir / Path(p).name
                if not dest.exists():
                    shutil.copy2(p, dest)
                copied += 1
                if copied >= 10:
                    break
            print(f"Placed DICOM demo in: {dicom_sample_dir}")
        else:
            print(
                "No public DICOM sources succeeded. For real DICOM, we recommend Kaggle SIIM-ACR Pneumothorax (requires Kaggle API).\n"
                "Steps:\n"
                "  1) pip install kaggle\n"
                "  2) Place kaggle.json at %USERPROFILE%/.kaggle/kaggle.json (Windows) or ~/.kaggle/kaggle.json\n"
                "  3) kaggle competitions download -c siim-acr-pneumothorax-segmentation -p data/dicom\n"
                "  4) Unzip train*.zip into data/dicom, then re-run this script to copy a small subset to data/inputs/dicom_sample\n"
            )
    except Exception as e:
        print(f"Post-copy to inputs failed (ignored): {e}")

    print("Done. Examples saved under 'data/nifti' and 'data/dicom'.")
    print(
        "Also prepared quick-start copies in 'data/inputs' (one NIfTI and a small DICOM series)."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed: {exc}")
        sys.exit(1)
