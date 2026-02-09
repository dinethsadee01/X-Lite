"""Check TorchXRayVision disease classes"""
import torchxrayvision as xrv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print('TorchXRayVision pathologies (18 total):')
for i, pathology in enumerate(xrv.datasets.default_pathologies):
    print(f'  [{i:2d}] {pathology}')

print(f'\nTotal: {len(xrv.datasets.default_pathologies)} diseases')

# Check which overlap with ChestX-ray14
from config.disease_labels import DISEASE_LABELS
print('\n\nOverlap with ChestX-ray14:')
xrv_set = set(xrv.datasets.default_pathologies)
chestxray14_set = set([d for d in DISEASE_LABELS if d != 'No_Finding'])

overlap = xrv_set.intersection(chestxray14_set)
print(f'Common diseases: {len(overlap)}/{len(chestxray14_set)}')
for disease in sorted(overlap):
    print(f'  ✓ {disease}')

print('\n\nChestX-ray14 diseases NOT in XRV:')
missing = chestxray14_set - xrv_set
for disease in sorted(missing):
    print(f'  ✗ {disease}')

print('\n\nXRV diseases NOT in ChestX-ray14:')
extra = xrv_set - chestxray14_set
for disease in sorted(extra):
    print(f'  + {disease}')
