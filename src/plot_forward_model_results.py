"""
plot the results of beta slopes from the forward modelling analysis.

Created: Tuesday 10th March 2026.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import glob

plt.rcParams.update({'font.size': 15})
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['figure.dpi'] = 100

plt.rcParams.update({
    # Ticks on all sides, pointing inwards
    'xtick.top': True, 'xtick.bottom': True,
    'ytick.left': True, 'ytick.right': True,
    'xtick.direction': 'in', 'ytick.direction': 'in',

    # Major tick size and width
    'xtick.major.size': 6.5, 'ytick.major.size': 6.5,
    'xtick.major.width': 2, 'ytick.major.width': 2,

    # Minor tick size and width
    'xtick.minor.size': 3, 'ytick.minor.size': 3,
    'xtick.minor.width': 1.5, 'ytick.minor.width': 1.5,
})

redshift_bin = 'z7'
field_name = 'COSMOS'

cat_dir = Path.cwd().parent / "data" / "catalogues"
cat_name = f"{field_name}_{redshift_bin}_real_errors.fits"

results_dir = Path.cwd() / f"uv_slope_{redshift_bin}_{field_name}_results_sn3_with_limits"

plot_dir = Path.cwd().parent / "plots"

t = Table.read(cat_dir / cat_name)
t.sort('Muv')

#! In the results dir, get all .txt files
txt_files = glob.glob(str(results_dir / "*.txt"))

# Results look like
# obj 0 z 6.78218
# n_candidates 5 n_detections 5
# used_filters: ['JE', 'HE', 'J', 'H', 'Ks']
# A = 5.32075e-19 +1.00892e-20 -9.96641e-21
# beta = -1.76031 +0.12373 -0.12465

# Extract beta values and errors, filters used, and Muv for each object.
# Muv will come from the catalogue, matched by row number (assuming the order is the same).
betas = []
beta_errs = []
used_filters = []
Muvs = []

for txt_file in txt_files:
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        # Extract beta and error from the line starting with "beta ="
        for line in lines:
            if line.startswith("beta ="):
                parts = line.split()
                beta = float(parts[2])
                beta_err_plus = float(parts[3][1:])  # +0.12373 -> 0.12373
                beta_err_minus = float(parts[4][1:])  # -0.12465 -> 0.12465
                betas.append(beta)
                beta_errs.append((beta_err_plus, beta_err_minus))
            elif line.startswith("used_filters:"):
                filters_str = line.split("used_filters:")[1].strip()
                filters_list = eval(filters_str)  # Convert string list to actual list
                used_filters.append(filters_list)
            elif line.startswith("obj"):
                # Extract object index and redshift (not used here but could be useful)
                pass

# Bin the results in Muv so that each bin has 10 objects, and compute the median beta and error in each bin.
bin_size = 10
binned_Muvs = []
binned_betas = []
binned_beta_errs = []

for i in range(0, len(betas), bin_size):
    bin_betas = betas[i:i+bin_size]
    bin_beta_errs = beta_errs[i:i+bin_size]
    bin_Muvs = t['Muv'][i:i+bin_size]  # Assuming the order of txt files matches the catalogue rows
    
    median_beta = np.median(bin_betas)
    median_Muv = np.median(bin_Muvs)
    
    # For error, we can take the median of the plus and minus errors separately
    median_beta_err_plus = np.median([err[0] for err in bin_beta_errs])
    median_beta_err_minus = np.median([err[1] for err in bin_beta_errs])

    # Or we could take the standard deviation of the betas in the bin as an error estimate
    # median_beta_err = np.std(bin_betas)
    # median_beta_err_minus = median_beta_err_plus = median_beta_err

    
    binned_Muvs.append(median_Muv)
    binned_betas.append(median_beta)
    binned_beta_errs.append((median_beta_err_plus, median_beta_err_minus))



# Plot beta vs Muv with error bars
plt.figure(figsize=(8, 6))
Muvs = t['Muv'][:len(betas)]  # Assuming the order of txt files matches the catalogue rows
beta_errs_plus = [err[0] for err in beta_errs]
beta_errs_minus = [err[1] for err in beta_errs]
plt.errorbar(Muvs, betas, yerr=[beta_errs_minus, beta_errs_plus], 
             fmt='o', color='black', alpha=0.4, markeredgecolor='none')
plt.xlabel(r'$M_{\rm UV}$')
plt.ylabel(r'$\beta$')

# Also plot the binned medians with error bars
binned_beta_errs_plus = [err[0] for err in binned_beta_errs]
binned_beta_errs_minus = [err[1] for err in binned_beta_errs]
plt.errorbar(binned_Muvs, binned_betas, yerr=[binned_beta_errs_minus, binned_beta_errs_plus], 
             fmt='o', color='red', elinewidth=3, label='Binned medians', markersize=15,
             markeredgecolor='black', capsize=4, zorder=10, capthick=3)

text_dict = {'z6': r'$z=6$', 'z7': r'$z=7$'}

# Add text annotation for the redshift bin
plt.text(0.05, 0.95, text_dict[redshift_bin], transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')

if redshift_bin == 'z7':
    # Control axes labels on x axis
    vals = [-22.5, -22, -21.5, -21, -20.5, -20]
    plt.xticks(vals)

plt.savefig(plot_dir / f"beta_vs_Muv_{redshift_bin}_{field_name}.pdf", dpi=300, bbox_inches='tight')




