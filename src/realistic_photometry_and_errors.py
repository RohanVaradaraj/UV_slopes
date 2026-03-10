"""
realistic_photometry_and_errors.py

Go back to the z=6-7 sources, and measure the actual statistical errors, without the 5% minimum error.

Created: Friday 27th February 2026.
"""

from dataclasses import field
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.wcs import WCS
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MplPath

#! SWITCHES
do_plot = True

#! Dictionary to deal with U+E column names to images.lis
filter_dict = {
    'HSC_G': 'HSC-G_DR3',
    'HSC_R': 'HSC-R_DR3',
    'HSC_I': 'HSC-I_DR3',
    'HSC_NB0816': 'HSC-NB0816_DR3',
    'HSC_Z': 'HSC-Z_DR3',
    'HSC_NB0921': 'HSC-NB0921_DR3',
    'HSC_Y': 'HSC-Y_DR3',
    'IE': 'VIS',
    'YE': 'Ye',
    'JE': 'Je',
    'HE': 'He',
    'Y': 'Y',
    'J': 'J',
    'H': 'H',
    'Ks': 'Ks',
    'IRAC1': 'ch1cds',
    'IRAC2': 'ch2cds'
}

def grid_depths(gridTable: dict, x: np.ndarray, y: np.ndarray, faster: bool = True, verbose: bool = False, nearby: bool = False) -> np.ndarray:
    ''' 
    Code to find the closest depth measurement from my previous analysis. Faster than truly local depths

    Parameters:
    gridTable (dict): Dictionary containing grid information.
    x (np.ndarray): Array of x coordinates.
    y (np.ndarray): Array of y coordinates.
    faster (bool, optional): If True, use a faster method. Defaults to True.
    verbose (bool, optional): If True, enable verbose output. Defaults to False.
    nearby (bool, optional): If True, consider nearby pixels. Defaults to False.

    Returns:
    np.ndarray: Array of depth values.
    '''

    xgrid = gridTable['x']
    ygrid = gridTable['y']
    keys = gridTable.colnames

    depthsOverField = gridTable['depths']

    ## Make an output array
    depthArray = np.zeros(x.size)
    depthArray[:] = -99.0

    if faster:
        if verbose:
            print("Using faster method.")
            print("Input array size is ", x.size)
        deltay = np.min(ygrid)
        deltax = np.min(xgrid)

    ## loop through the grid instead of each object
        for xi in range(xgrid.size):
            
            xmin = xgrid[xi] - deltax
            xmax = xgrid[xi] + deltax
            ymin = ygrid[xi] - deltay
            ymax = ygrid[xi] + deltay

            ii = (x > xmin) & (x <= xmax) & (y > ymin) & (y <= ymax)
            
            depthArray[ii] = depthsOverField[xi]

    else:
        
    ## Find the closest point to the objects x and y positions
    ## Loop!
        for xi in range(x.size):
            
        ## make a radius array
            deltax = (xgrid - x[xi])
            deltay = (ygrid - y[xi])
            radius = np.sqrt(deltax*deltax + deltay*deltay)
            mini = np.argmin(radius)
            
        ## try using argpartition
            numpoints = 10
            idx = np.argpartition(radius, numpoints)
            
            if nearby:
                
                mini = idx[0:numpoints]
                print("The nearby depths are = ", depthsOverField[mini])
                print("Before = ", depthsOverField[mini][0])
            

            depthArray[xi] = depthsOverField[mini][0]

    return depthArray
        

def read_polygon_from_txt(path):
    # File is as string, with each line as 'ra, dec' of the four corners
    polygon = []
    with open(path, 'r') as f:
        for line in f:
            ra, dec = map(float, line.strip().split(','))
            polygon.append((ra, dec))
    return polygon


def points_in_polygon_mask(ra_arr, dec_arr, polygon):
    pts = np.column_stack((ra_arr, dec_arr))
    path = MplPath(polygon)  # expects sequence of (x,y) == (ra,dec)
    return path.contains_points(pts)


def min_dist_to_edges(points, poly):
    A = np.asarray(poly, dtype=float)     
    B = np.vstack([A[1:], A[0]])
    AB = B - A
    P = np.asarray(points, dtype=float)[:, None, :]
    A_ = A[None, :, :]
    AB_ = AB[None, :, :]
    AP = P - A_
    t = np.clip(np.sum(AP * AB_, axis=2) / np.sum(AB_ * AB_, axis=2), 0, 1)
    closest = A_ + t[..., None] * AB_
    return np.sqrt(np.min(np.sum((P - closest)**2, axis=2), axis=1))

# Set up directories
cat_dir = Path.cwd().parents[1] / 'euclid' / 'data' / 'catalogues' / 'candidates'
euclid_cat_dir = Path.cwd().parents[1] / 'euclid' / 'src' / 'paper_corrections'
output_dir = Path.cwd().parent / 'data' / 'catalogues'
footprint_dir = Path.cwd().parent / 'data' / 'footprints'
data_dir = Path.cwd().parents[2] / 'data'
depth_dir = Path.cwd().parents[2] / 'data' / 'depths'

field_names = ['COSMOS']
redshift_bins = ['z7']

# Dictionary of field names to candidate catalogues, with redshift keys inside for z6 and z7.
catalogues = {'COSMOS': {'z6': cat_dir / 'COSMOS_5sig_HSC_Z_nonDet_HSC_G_nonDet_HSC_R_candidates_2025_06_06.fits', 
                         'z7': euclid_cat_dir / 'UltraVISTA_plus_Euclid_z7_sample.fits'},
                'XMM': {'z6': cat_dir / 'XMM_5sig_HSC_Z_nonDet_HSC_G_nonDet_HSC_R_candidates_2025_05_14.fits',
                        'z7': None}
            }


for field_name in field_names:
    for redshift_bin in redshift_bins:
        print(f'Running for {field_name} {redshift_bin}')

        # Load the catalogue
        cat_file = catalogues[field_name][redshift_bin]
        if cat_file is None:
            print(f'No catalogue found for {field_name} {redshift_bin}, skipping')
            continue
        cat = Table.read(cat_file)

        ra = cat['RA']
        dec = cat['DEC']

        #! IF WE ARE NOT IN COSMOS, DEAL WITH THE VIDEO TILES
        if field_name != 'COSMOS':
            if field_name == 'XMM':
                tile1 = read_polygon_from_txt(footprint_dir / 'XMM1_footprint.txt')
                tile2 = read_polygon_from_txt(footprint_dir / 'XMM2_footprint.txt')
                tile3 = read_polygon_from_txt(footprint_dir / 'XMM3_footprint.txt')

            if field_name == 'CDFS':
                tile1 = read_polygon_from_txt(footprint_dir / 'CDFS1_footprint.txt')
                tile2 = read_polygon_from_txt(footprint_dir / 'CDFS2_footprint.txt')
                tile3 = read_polygon_from_txt(footprint_dir / 'CDFS3_footprint.txt')

            tiles = [tile1, tile2, tile3]
            
            # Create a mask for each tile
            masks = [points_in_polygon_mask(ra, dec, tile) for tile in tiles]
            print(f'Masks for {field_name} {redshift_bin}: {[np.sum(mask) for mask in masks]}')

            # Where there are duplicates, find distance to edge in both tiles and choose one with largest distance to edge
            combined_mask = np.sum(masks, axis=0)

            # Find indices of sources in multiple tiles
            duplicate_indices = np.where(combined_mask > 1)[0]
            print(f'Found {len(duplicate_indices)} duplicate sources in {field_name} {redshift_bin}')

            stack = np.vstack(masks)                  # (ntile, N)
            combined_mask = np.sum(stack, axis=0)

            # resolve duplicates
            pts_dup = np.column_stack((ra[duplicate_indices], dec[duplicate_indices]))
            dists = np.array([min_dist_to_edges(pts_dup, tile) for tile in tiles])  # (ntile, Ndup)

            # mask out tiles that don't actually contain the source
            dists[~stack[:, duplicate_indices]] = -np.inf

            best_tile = np.argmax(dists, axis=0)

            # remove duplicates from all tiles
            for i in range(len(tiles)):
                masks[i][duplicate_indices] = False

            # reassign to best tile
            for j, idx in enumerate(duplicate_indices):
                masks[best_tile[j]][idx] = True
  
            #! Plot tiles and sources, with sources coloured by their tile assignment, to check
            if do_plot:
                plt.figure(figsize=(12,6))
                colors = ['red', 'green', 'blue']

                for i, tile in enumerate(tiles):
                    tile_arr = np.asarray(tile)

                    # close polygon
                    tile_closed = np.vstack([tile_arr, tile_arr[0]])

                    plt.plot(tile_closed[:,0], tile_closed[:,1],
                            color=colors[i], label=f'Tile {i+1}')

                    plt.scatter(ra[masks[i]], dec[masks[i]],
                                color=colors[i], s=10, alpha=0.5)

                plt.xlabel('RA')
                plt.ylabel('Dec')
                plt.title(f'{field_name} {redshift_bin} Tile Assignments')
                plt.legend()
                plt.gca().invert_xaxis()
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig(Path.cwd().parent / 'plots' / f'{field_name}_{redshift_bin}_footprint_and_candidates.pdf')
                plt.show()

            # Find all columns with 'flux_' in the name, and get the filt from flux_{filt}
            flux_cols = [col for col in cat.colnames if 'flux_' in col]
            filters = [col.split('flux_')[1] for col in flux_cols]
            print(f'Found filters: {filters}')

            # create empty columns in full catalogue
            for filt in filters:
                colname = f'err_real_{filt}'
                if colname not in cat.colnames:
                    cat[colname] = np.full(len(cat), np.nan)

            #! Now loop through the tiles
            errors_tiles = []
            for i, tile in enumerate(tiles):
                print(f'Processing {field_name} {redshift_bin} Tile {i+1}')

                # Select sources in this tile
                mask = masks[i]
                cat_tile = cat[mask]

                # Open the images.list file for this tile, and find the image for each filter, commented header
                info = Table.read(data_dir / f'{field_name}{i+1}' / 'images.lis', format='ascii.commented_header')

                good_filters = filters.copy()
                for j, filt in enumerate(filters):
                    print(f'Processing filter {filt}')

                    # Find the image for this filter
                    img_row = info[info['Name'] == filt]
                    if len(img_row) == 0:
                        print(f'No image found for filter {filt} in {field_name} Tile {i+1}, skipping')
                        # Remove this filter from the list of filters to process for this tile
                        good_filters.remove(filt)
                        continue

                    img_path = img_row['directory'][0]
                    print(f'Found image: {img_path}')

                    if img_path == 'here':
                        img_dir = data_dir / f'{field_name}{i+1}'
                    else:
                        img_dir = Path(img_path)

                    img_file = img_dir / img_row['Image'][0]

                    with fits.open(img_file) as hdul:
                        img_header = hdul[0].header
                        wcs = WCS(img_header)

                    # For all the sources in the mask for this tile, get pixel coords
                    x, y = wcs.world_to_pixel_values(cat_tile['RA'], cat_tile['DEC'])

                    # Get depth table for this tile and filter
                    depth_table = Table.read(depth_dir / f'{field_name}{i+1}' / 'phot' / f'{filt}_2.0as_gridDepths_300_200.fits')

                    depths = grid_depths(depth_table, x, y)

                    #! Now we have the depths, we can calculate the errors on the fluxes, and add to the catalogue

                    value = -(48.6 + depths)/2.5
                    errors = 0.2*(10**value)

                    errors_tiles.append(errors)

                    # Add these errors to the catalogue as a new column, with name err_real_{filt}
                    cat[f'err_real_{filt}'][mask] = errors

                # Plot histogram of errors for this tile, in log space, for all filters combined
                if do_plot:
                    plt.figure(figsize=(8,6))
                    plt.hist(np.log10(np.concatenate(errors_tiles)), bins=100, alpha=0.7, label='Actual Errors', density=True)
                    plt.xlabel('log10(err)')
                    plt.ylabel('Relative Number of Sources')
                    plt.title(f'{field_name} {redshift_bin} Tile {i+1} flux errors')

                    # Also plot errors from the catalogue, if they exist, for comparison
                    errors_cat = []
                    for j, filt in enumerate(good_filters):
                        error_col = f'err_{filt}'
                        if error_col in cat_tile.colnames:
                            errors_cat.append(cat_tile[error_col])
                    
                    # Remove all <0 errors from the catalogue errors
                    errors_cat = [err for err in errors_cat if np.all(err > 0)]
                    if len(errors_cat) > 0:
                        plt.hist(np.log10(np.concatenate(errors_cat)), bins=100, alpha=0.7, label='Catalogue Errors', density=True)
                        plt.legend()

                    plt.savefig(Path.cwd().parent / 'plots' / f'{field_name}_{redshift_bin}_Tile{i+1}_flux_error_histogram.pdf')
                    plt.show()

                output_file = output_dir / f'{field_name}_{redshift_bin}_real_errors.fits'
                cat.write(output_file, overwrite=True)
                print(f'Saved updated catalogue to {output_file}')

        if field_name == 'COSMOS':
            # For COSMOS, we can just calculate the depths at the positions of the sources, without worrying about tiles
            print(f'Processing {field_name} {redshift_bin}')

            # Find all columns with 'flux_' in the name, and get the filt from flux_{filt}
            flux_cols = [col for col in cat.colnames if 'flux_' in col]
            filters = [col.split('flux_')[1] for col in flux_cols]
            print(f'Found filters: {filters}')

            # create empty columns in full catalogue
            good_filters = filters.copy()
            for filt in filters:
                colname = f'err_real_{filt}'
                if colname not in cat.colnames:
                    cat[colname] = np.full(len(cat), np.nan)

            # Open the image for each filter, commented header
            info = Table.read(data_dir / f'{field_name}' / 'images.lis', format='ascii.commented_header')

            for j, filt in enumerate(filters):
                print(f'Processing filter {filt}')

                if redshift_bin == 'z7':
                    # Use the dict to convert from the U+E filter name to the image.lis name
                    filter_name = filter_dict[filt]

                # Find the image for this filter
                img_row = info[info['Name'] == filter_name]
                if len(img_row) == 0:
                    print(f'No image found for filter {filter_name} in {field_name}, skipping')
                    # Remove from good filters
                    good_filters.remove(filt)
                    continue

                img_path = img_row['directory'][0]
                print(f'Found image: {img_path}')

                if img_path == 'here':
                    img_dir = data_dir / f'{field_name}'
                else:
                    img_dir = Path(img_path)

                img_file = img_dir / img_row['Image'][0]

                with fits.open(img_file) as hdul:
                    img_header = hdul[0].header
                    wcs = WCS(img_header)

                # Get pixel coords for all sources
                x, y = wcs.world_to_pixel_values(ra, dec)

                # Get depth table for this filter
                depth_table = Table.read(depth_dir / f'{field_name}' / 'phot' / f'{filter_name}_2.0as_gridDepths_300_200.fits')

                depths = grid_depths(depth_table, x, y)

                # Now we have the depths, we can calculate the errors on the fluxes, and add to the catalogue

                value = -(48.6 + depths)/2.5
                errors = 0.2*(10**value)

                cat[f'err_real_{filt}'] = errors

            # Plot histogram of errors for this field and redshift bin, in log space, for all filters combined
            if do_plot:
                plt.figure(figsize=(8,6))

                all_errors = np.concatenate([cat[f'err_real_{filt}'] for filt in filters])
                # Truncate errors with log(err) < -28
                all_errors = all_errors[all_errors < 1e-28]
                plt.hist(np.log10(all_errors), bins=100, alpha=0.7, label='Actual Errors', density=True)
                plt.xlabel('log10(err)')
                plt.ylabel('Relative number of Sources')
                plt.title(f'{field_name} {redshift_bin} flux errors')

                # alSO plot positive errors from the catalogue, if they exist, for comparison
                errors_cat = []
                for j, filt in enumerate(good_filters):
                    error_col = f'err_{filt}'
                    if error_col in cat.colnames:
                        errors_cat.append(cat[error_col])
                # Remove all <0 errors from the catalogue errors
                errors_cat = [err for err in errors_cat if np.all(err > 0)]
                # Also truncate catalogue errors with log(err) < -28
                errors_cat = [err for err in errors_cat if np.all(err < 1e-28)]
                if len(errors_cat) > 0:
                    plt.hist(np.log10(np.concatenate(errors_cat)), bins=100, alpha=0.7, label='Catalogue Errors', color='orange', density=True)

                plt.legend()
                plt.savefig(Path.cwd().parent / 'plots' / f'{field_name}_{redshift_bin}_flux_error_histogram.pdf')
                plt.show()

            output_file = output_dir / f'{field_name}_{redshift_bin}_real_errors.fits'
            cat.write(output_file, overwrite=True)
            print(f'Saved updated catalogue to {output_file}')


            


            


