
import sys
from pathlib import Path
import numpy as np
if not hasattr(np, 'asscalar'):
    np.asscalar = np.ndarray.item
if not hasattr(np, 'alen'):
    def alen(a):
        return len(a)
    np.alen = alen
from astropy.table import Table

REPO_ROOT = Path("/home/mishran/Documents/VAPORS")
sys.path.insert(0, str(REPO_ROOT))

from cloudy_voigt_inference import (
    JointCloudyComponentFitter,
    CloudyGridInterpolator,
    FitterConfig,
    KDEObservation,
)

# Config
CHAIN_PATH = REPO_ROOT / "chains/New_J2135_z0.57_93_2_contamination_chain.npy"
PARAM_NAMES_PATH = REPO_ROOT / "chains/New_J2135_z0.57_93_2_contamination_param_names.npy"
GRID_PATH = Path('/data/mishran/cloudy_outputs/HM12/J2135_example/full_grid.dat')
COMPONENT_IDS = [0, 1]
IONS_PER_COMPONENT = [["N_HI", "N_CIII", "N_CIV", "N_OVI"], ["N_HI", "N_CIII", "N_CIV", "N_OVI"]]

# Load KDE Stats
chain = np.load(CHAIN_PATH)
param_names = np.load(PARAM_NAMES_PATH)
kde_samples = []
for comp_id, ion_list in zip(COMPONENT_IDS, IONS_PER_COMPONENT):
    for ion in ion_list:
        target = f"{ion}_{comp_id}"
        idx = list(param_names).index(target)
        kde_samples.append(np.log10(chain[:, idx] * 1e8))
kde_samples = np.array(kde_samples).T
kde_mean = np.mean(kde_samples, axis=0)
print(f"Target Mean (Log N): {kde_mean}")

# Load Grid
grid_table = Table.read(GRID_PATH, format="ascii")
renames = {"HI": "N_HI", "CII": "N_CII", "CIII": "N_CIII", "CIV": "N_CIV", "OVI": "N_OVI", "SiIV": "N_SiIV"}
for s, t in renames.items():
    if s in grid_table.colnames and t not in grid_table.colnames:
        grid_table.rename_column(s, t)
if "N_H" in grid_table.colnames:
    grid_table["N_H"] = np.round(np.log10(grid_table["N_H"]), 2)
grid = CloudyGridInterpolator.from_table(grid_table)

bounds = grid.parameter_bounds()
print(f"Grid Bounds: {bounds}")

# Check Max/Min possible values in Grid for target ions
for ion in ["N_HI", "N_CIII", "N_CIV", "N_OVI"]:
    vals = np.log10(grid_table[ion])
    print(f"{ion}: Grid Range [{np.min(vals):.2f}, {np.max(vals):.2f}]")
