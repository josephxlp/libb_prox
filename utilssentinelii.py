from glob import glob
import rasterio
import numpy as np


import os
import rasterio

def split_raster_by_bands(input_file, bandnames):
    bandfiles = []

    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    # Read the input file
    with rasterio.open(input_file) as src:
        # Check if the number of bands matches
        if len(bandnames) != src.count:
            raise ValueError("Number of bands in the file does not match the bandnames list.")

        # Check for existing files for the top 5 bands
        top_5_files = [input_file.replace("S2RGB", f"S2_{bandname}") for bandname in bandnames[:5]]
        if all(os.path.exists(file) for file in top_5_files):
            print("Top 5 files already exist. Skipping processing.")
            return top_5_files

        # Save each band as a separate file
        for idx, bandname in enumerate(bandnames, start=1):
            try:
                output_file = input_file.replace("S2RGB", f"S2_{str(bandname).upper()}")
            except ValueError:
                output_file = input_file.replace("S2", f"S2_{str(bandname).upper()}")

            bandfiles.append(output_file)

            # Read the current band
            band_data = src.read(idx)

            # Create a new file for the current band
            profile = src.profile
            profile.update(
                count=1,  # Single band
                dtype=band_data.dtype
            )

            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(band_data, 1)

            print(f"Saved: {output_file}")

    return bandfiles


def get_indices(file_paths):
    """
    Compute spectral indices and save them as individual raster files.

    Parameters:
        file_paths (list): List of file paths for the bands.
    """
    # Load bands
    bands = {}
    with rasterio.open(file_paths[0]) as src:
        profile = src.profile
        profile.update(count=1)

    bands['RED'] = rasterio.open(file_paths[0]).read(1).astype(np.float32)
    bands['GREEN'] = rasterio.open(file_paths[1]).read(1).astype(np.float32)
    bands['BLUE'] = rasterio.open(file_paths[2]).read(1).astype(np.float32)
    bands['NIR'] = rasterio.open(file_paths[3]).read(1).astype(np.float32)
    bands['SWIR1'] = rasterio.open(file_paths[4]).read(1).astype(np.float32)

    # Compute indices
    indices = {
        'NDVI': (bands['NIR'] - bands['RED']) / (bands['NIR'] + bands['RED']),
        'ENDVI': (bands['NIR'] + bands['GREEN'] - 2 * bands['RED']) / (bands['NIR'] + bands['GREEN'] + 2 * bands['RED']),
        'NDWI': (bands['GREEN'] - bands['NIR']) / (bands['GREEN'] + bands['NIR']),
        'ANDWI': (bands['GREEN'] - bands['SWIR1']) / (bands['GREEN'] + bands['SWIR1']),
        'NBAI': (bands['SWIR1'] - bands['RED']) / (bands['SWIR1'] + bands['RED']),
        'UI': (bands['SWIR1'] - bands['NIR']) / (bands['SWIR1'] + bands['NIR']),
    }

    # Save indices
    for name, data in indices.items():
        output_file = file_paths[0].replace("_red.tif", f"_{name}.tif")
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(np.nan_to_num(data), 1)
        print(f"Saved: {output_file}")