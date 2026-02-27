"""
Script to grab the footprints of the VISTA tiles, to check the positions of sources within them.

Created: Friday 27th February 2026.
"""

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

# Don't need for COSMOS

data_dir = Path.cwd().parents[2] / 'data'

#! XMM
xmm1 = data_dir / 'XMM1' / 'HSC-G_DR3.fits'
xmm2 = data_dir / 'XMM2' / 'HSC-G_DR3.fits'
xmm3 = data_dir / 'XMM3' / 'HSC-G_DR3.fits'

names = ['XMM1', 'XMM2', 'XMM3']

# Open these fits files and save the footprints
for i, xmm in enumerate([xmm1, xmm2, xmm3]):
    with fits.open(xmm) as hdul:
        wcs = WCS(hdul[0].header)
        footprint = wcs.calc_footprint()
        print(f'Footprint for {xmm.name}: {footprint}')

        # Save footprint to ../data/footprints
        output_file = Path.cwd().parent / 'data' / 'footprints' / f'{names[i]}_footprint.txt'
        with open(output_file, 'w') as f:
            for row in footprint:
                f.write(f'{row[0]}, {row[1]}\n')
        print(f'Saved footprint for {xmm.name} to {output_file}')

#! Same for CDFS
cdfs1 = data_dir / 'CDFS1' / 'HSC-G.fits'
cdfs2 = data_dir / 'CDFS2' / 'HSC-G.fits'
cdfs3 = data_dir / 'CDFS3' / 'HSC-G.fits'
names = ['CDFS1', 'CDFS2', 'CDFS3']

for i, cdfs in enumerate([cdfs1, cdfs2, cdfs3]):
    with fits.open(cdfs) as hdul:
        wcs = WCS(hdul[0].header)
        footprint = wcs.calc_footprint()
        print(f'Footprint for {cdfs.name}: {footprint}')

        # Save footprint to ../data/footprints
        output_file = Path.cwd().parent / 'data' / 'footprints' / f'{names[i]}_footprint.txt'
        with open(output_file, 'w') as f:
            for row in footprint:
                f.write(f'{row[0]}, {row[1]}\n')
        print(f'Saved footprint for {cdfs.name} to {output_file}')
