import nibabel as nib
import numpy as np

mask_path = "scripts/data/abdomen/labelsTr/label0005.nii.gz"

mask = nib.load(mask_path).get_fdata()
vals, counts = np.unique(mask.astype(np.int32), return_counts=True)

print("unique labels:", vals)
print("num labels:", len(vals))
for v, c in zip(vals, counts):
    print(v, c)