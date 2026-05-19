# Med-MV-SAM3D

Training-free medical 3D reconstruction scaffold built around SAM3D-style shape proposals, multi-plane slice fusion, and anatomical prior refinement.

The first project stage focuses on running the full pipeline with real NIfTI data when available and a simulated SAM3D proposal generator while the real SAM3D environment is being configured.

## Pipeline

1. Read a medical scan and segmentation mask from NIfTI.
2. Normalize CT/MRI intensities and extract target-organ masks.
3. Select top-k axial, coronal, and sagittal slices by mask area.
4. Encode each slice as pseudo-RGB input for SAM3D.
5. Generate SAM3D candidate point clouds.
6. Fuse candidates with union, weighted, or silhouette-constrained fusion.
7. Apply anatomical post-processing: largest component, hole filling, closing, smoothing, and optional volume prior correction.
8. Evaluate against ground truth with Dice, voxel IoU, voxel Dice, F1@0.01, Chamfer Distance, EMD, HD95, ASD, and relative volume error.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the synthetic smoke-test demo:

```bash
python run_demo.py --output outputs/demo_case
```

The demo writes extracted slice images, simulated SAM3D candidates, fused/refined outputs, and `metrics.csv`.

## Real Data Layout

Recommended layout:

```text
data/
  raw/
    BTCV/
  processed/
outputs/
  candidates/
  fused/
  refined/
  metrics/
  figures/
```

Example preprocessing:

```bash
python scripts/01_preprocess_nifti.py \
  --scan data/raw/BTCV/case001_img.nii.gz \
  --mask data/raw/BTCV/case001_seg.nii.gz \
  --case-id case001 \
  --organ liver \
  --label 6 \
  --output data/processed/case001_liver
```

Extract multi-plane slices:

```bash
python scripts/02_extract_multiplane_slices.py \
  --processed data/processed/case001_liver \
  --k 3
```

## SAM3D Integration Point

`medmvsam3d.sam3d_runner.run_sam3d_stub` is a placeholder. Replace it with a real call in `run_sam3d` once the SAM3D environment is ready. Keep the output contract stable: each 2D image/mask pair should produce a `.ply` point cloud or mesh-sampled point cloud.

