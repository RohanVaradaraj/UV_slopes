"""
Fit power law slopes to the real photometry.

Created: Tuesday 10th March 2026.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from pathlib import Path

#! Directories
cat_dir = Path.cwd().parent / 'data' / 'catalogues'
filter_dir = Path.cwd().parent / 'data' / 'filters'

#! Load catalogue
t = Table.read(cat_dir / 'COSMOS_z6_real_errors.fits')

# Sort by Muv
t.sort('Muv')

t = t[0:10]
# print(t.colnames)

# Dictionary for filter names to filter transmission curve file names
filter_dict = {'HSC-G_DR3': 'g_HSC.txt',
               'HSC-R_DR3': 'r_HSC.txt',
               'HSC-I_DR3': 'i_HSC.txt',
               'HSC-NB0816_DR3': 'nb816_HSC.txt',
               'HSC-Z_DR3': 'z_HSC.txt',
               'HSC-NB0921_DR3': 'nb921_HSC.txt',
               'HSC-Y_DR3': 'y_HSC.txt',
               'Y': 'VISTA_Y.txt',
               'J': 'VISTA_J.txt',
               'H': 'VISTA_H.txt',
               'Ks': 'VISTA_Ks.txt',
               'f115w': 'f115w_angstroms.txt',
               'f150w': 'f150w_angstroms.txt',
               'f277w': 'f277w_angstroms.txt',
               'f444w': 'f444w_angstroms.txt',
               'VIS': 'Euclid_VIS.txt',
               'Ye': 'Euclid_Y.txt',
               'Je': 'Euclid_J.txt',
               'He': 'Euclid_H.txt',
               'ch1cds': 'irac_ch1.txt',
                'ch2cds': 'irac_ch2.txt',
               }

#! From the column names which go 'flux_{filter_name}', get all available filters
flux_cols = [col for col in t.colnames if col.startswith('flux_')]
filters = [col.split('flux_')[1] for col in flux_cols]
print(filters)

for i, obj in enumerate(t):

    # Get the object redshift
    z = obj['Zphot']

    # Collect fluxes and errors for all filters
    fluxes = np.array([obj[f'flux_{filter_name}'] for filter_name in filters])
    errors = np.array([obj[f'err_real_{filter_name}'] for filter_name in filters])

    # Get central wavelengths of filters    
    central_wavelengths = []
    for filter_name in filters:
        filter_file = filter_dir / filter_dict[filter_name]
        filter_data = np.loadtxt(filter_file)
        central_wavelength = np.sum(filter_data[:, 0] * filter_data[:, 1]) / np.sum(filter_data[:, 1]) # Weighted average of wavelengths by transmission
        central_wavelengths.append(central_wavelength)

    # Find the filters which lie entirely in the range 1500 - 3000 Angstroms in the rest frame
    selected_filters = []
    selected_fluxes = []
    selected_errors = []
    for filter_name in filters:
        filter_file = filter_dir / filter_dict[filter_name]
        filter_data = np.loadtxt(filter_file)
        filter_wavelengths = filter_data[:, 0] # Assuming the first column is wavelength in Angstroms

        rest_frame_wavelengths = filter_wavelengths / (1 + z)

        # Enforce 95% of filter area to be in the 1500-3000 range
        filter_area = np.trapz(filter_data[:, 1], filter_wavelengths) # Assuming the second column is transmission
        in_range_area = np.trapz(filter_data[(rest_frame_wavelengths >= 1500) & (rest_frame_wavelengths <= 3000), 1], filter_wavelengths[(rest_frame_wavelengths >= 1500) & (rest_frame_wavelengths <= 3000)])
        if in_range_area / filter_area >= 0.5:
            selected_filters.append(filter_name)
            selected_fluxes.append(obj[f'flux_{filter_name}'])
            selected_errors.append(obj[f'err_real_{filter_name}'])

    # plt.savefig('phot_and_filters.pdf')


    
    print(f'Object {i} at z={z:.2f} has selected filters: {selected_filters}')

    # Plot the photometry and filters of the selected filters, to check the selection is working as intended.
    plt.figure(figsize=(10, 6))
    plt.errorbar(central_wavelengths, fluxes, yerr=errors, fmt='o', label='All filters')
    plt.errorbar([central_wavelengths[filters.index(f)] for f in selected_filters], selected_fluxes, yerr=selected_errors, fmt='o', label='Selected filters', color='red')
    plt.yscale('log')
    plt.ylim(1e-31, 1e-28)

    # Add horizontal lines at 1e-29 for the observed wavelengths corresponding to 1500 and 3000 Angstroms in the rest frame, to check the selection is working as intended.
    plt.axvline(1500 * (1 + z), color='grey', linestyle='--')
    plt.axvline(3000 * (1 + z), color='black', linestyle='--')
    plt.show()
