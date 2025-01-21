import os
import rasterio
import numpy as np
from scipy import ndimage
from osgeo import gdal


def filter_tandemx_noise(dem_file, hem_file, com_file, n_iter=1):
    """Process DEM data by applying masks for noise and write the result to a new file."""

    # Define output file paths
    fdem_file = dem_file.replace('.tif', '_F.tif')
    mask_file = dem_file.replace('.tif', '_M.tif')

    gdal.UseExceptions()
    print('Loading and preprocessing DEM')

    # Open and read DEM file
    ds = open_ds(dem_file)
    if ds is None:
        return

    band = get_bands(ds, 1)
    dem = band_get_masked_array(band)
    print(f"DEM valid pixel count: {dem.count()}")
    mask = np.ma.getmaskarray(dem)

    print('Loading HEM and COM')
    hem = file_get_masked_array(hem_file, 1)
    com = file_get_masked_array(com_file, 1)

    if hem is None or com is None:
        return

    # Set up mask for invalid pixels based on HEM and COM
    max_err_multi = 1.5
    mask = np.logical_or(mask, (hem.data > max_err_multi))
    com_invalid = (0, 1, 2)
    mask = np.logical_or(mask, np.isin(com.data, com_invalid))

    print('Applying Masks')
    # Apply mask and dilate/erode to refine it
    dem_masked = np.ma.array(dem, mask=mask)
    mask = ndimage.binary_dilation(mask, iterations=n_iter)
    mask = ndimage.binary_erosion(mask, iterations=n_iter)

    # Write the mask and processed DEM to new files
    write_mask_as_geotif(dem_file, mask, mask_file)
    dem_masked = np.ma.array(dem, mask=mask)
    ndvalue = get_band_ndv(band)
    write_geotif(dem_file, fdem_file, dem_masked, ndvalue)

    return fdem_file, mask_file



def get_band_ndv(band):
    """Determine the NoData value for a raster band."""
    # Get NoData value
    ndv = band.GetNoDataValue()
    
    # If NoData is not defined, infer it based on the data range
    if ndv is None:
        data_array = band.ReadAsArray()
        max_val = np.max(data_array)
        min_val = np.min(data_array)

        # If minimum value is negative and below -9999, set NoData as the minimum value
        if min_val < 0 and min_val <= -9999:
            ndv = min_val
        # If maximum value exceeds absolute value of -9999, set NoData as the maximum value
        elif max_val > abs(-9999.):
            ndv = max_val
    
    return ndv

def band_get_masked_array(band):
    """Return a masked array of the raster band."""
    ndv = get_band_ndv(band)
    return np.ma.masked_values(band.ReadAsArray(), ndv)


def open_ds(path):
    """Open a dataset in read-only mode using GDAL."""
    try:
        return gdal.Open(path, gdal.GA_ReadOnly)
    except RuntimeError as e:
        print(f"Error opening file {path}: {e}")
        return None

def get_bands(ds, nbands):
    """Retrieve a specific raster band from the dataset."""
    return ds.GetRasterBand(nbands)


def file_get_masked_array(path, nbands):
    """Open a dataset and return a masked array for a specified band."""
    ds = open_ds(path)
    if ds is None:
        return None
    band = get_bands(ds, nbands)
    return band_get_masked_array(band)

def write_mask_as_geotif(input_file, mask, output_file):
    """Write the mask array to a new GeoTIFF file as uint8."""
    ds = open_ds(input_file)
    if ds is None:
        return

    # Remove existing file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Create the new dataset and write the mask
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_file, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(ds.GetGeoTransform())
    dataset.SetProjection(ds.GetProjection())

    band = dataset.GetRasterBand(1)
    band.WriteArray(mask.astype(np.uint8))
    dataset = None  # Close the dataset

def write_geotif(input_file, output_file, data_array, ndvalue=-9999.):
    """Write a masked array to a new GeoTIFF file."""
    ds = open_ds(input_file)
    if ds is None:
        return

    # Remove existing file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    data_array.set_fill_value(ndvalue)
    driver = gdal.GetDriverByName('GTiff')
    
    # Create the new dataset
    dataset = driver.Create(
        output_file, ds.RasterXSize, ds.RasterYSize, 
        ds.RasterCount, ds.GetRasterBand(1).DataType
    )
    dataset.SetMetadata(ds.GetMetadata())
    dataset.SetGeoTransform(ds.GetGeoTransform())
    dataset.SetProjection(ds.GetProjection())

    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(ndvalue)
    band.WriteArray(data_array.filled())
    dataset = None  # Close the dataset

def classify_lwm_TanDEMX_LCM(lcm_fn, lwm_a_fn, lwm_b_fn):
    with rasterio.open(lcm_fn) as src:
        b1 = src.read(1)
        b2 = src.read(2)
        
        # Classification based on conditions
        lwm_b1 = np.where(b1 == 3, 1, 0)
        lwm_b2 = np.where((b2 == 1) | (b2 == 2), 1, 0)
        
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1)
        
        # Writing the output for LWM A
        with rasterio.open(lwm_a_fn, 'w', **meta) as dst:
            dst.write(lwm_b1, 1)

        # Writing the output for LWM B
        with rasterio.open(lwm_b_fn, 'w', **meta) as dst:
            dst.write(lwm_b2, 1)
    
    return lwm_a_fn, lwm_b_fn

def classify_lwm_ESAWC(fipath, fopath, water_code=80):
    with rasterio.open(fipath) as src:
        data = src.read(1)
        
        # Classify based on the water_code
        lwm = (data == water_code).astype(np.uint8)
        
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1)
        
        # Write the output classification
        with rasterio.open(fopath, 'w', **meta) as dst:
            dst.write(lwm, 1)

def classify_lwm_CopWBM(fipath, fopath):
    # Define water classes (Ocean, Lake, River)
    water_classes = {1, 2, 3}
    
    with rasterio.open(fipath) as src:
        data = src.read(1)
        
        # Classify based on water_classes
        lwm = np.isin(data, list(water_classes)).astype(np.uint8)
        
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1)
        
        # Write the output classification
        with rasterio.open(fopath, 'w', **meta) as dst:
            dst.write(lwm, 1)


def classify_lwm_TanDEMX_WAM(fipath, fopath, lthreshold=33, hthreshold=127):
    """
    Classifies pixels in the TanDEM-X WAM into 'water' and 'non-water'
    and saves the result as a binary raster.

    Parameters:
    - fipath: str, path to the input TanDEM-X WAM dataset (GeoTIFF).
    - fopath: str, path to save the output binary raster (GeoTIFF).
    - lthreshold: int, minimum threshold value for water detection (default is 33).
    - hthreshold: int, maximum threshold value for water detection (default is 127).
    """ 
    with rasterio.open(fipath) as src:
        data = src.read(1)
        
        # Classify based on the threshold values
        lwm = ((data >= lthreshold) & (data <= hthreshold)).astype(np.uint8)
        
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1)
        
        # Write the output classification
        with rasterio.open(fopath, 'w', **meta) as dst:
            dst.write(lwm, 1)


def filter_water(fdem_file, mask_file, lcm_lwm_fn, esa_lwm_fn, wbm_lwm_fn):
    """Filters the DEM based on various mask layers and saves the result."""

    # Open the DEM file
    with rasterio.open(fdem_file) as dem_src:
        dem_data = dem_src.read(1)  # Read the first band
        dem_meta = dem_src.meta  # Get metadata for writing output

    # Open the mask file
    with rasterio.open(mask_file) as mask_src:
        mask_data = mask_src.read(1)  # Read the first band

    # Open additional mask files and combine them
    with rasterio.open(lcm_lwm_fn) as lcm_src:
        lcm_data = lcm_src.read(1)

    with rasterio.open(esa_lwm_fn) as esa_src:
        esa_data = esa_src.read(1)

    with rasterio.open(wbm_lwm_fn) as wbm_src:
        wbm_data = wbm_src.read(1)

    # Combine all masks: keep DEM values where all masks are 0
    combined_mask = (mask_data == 0) & (lcm_data == 0) & (esa_data == 0) & (wbm_data == 0)
    final_dem = np.where(combined_mask, dem_data, np.nan)  # Use np.nan for no-data

    # Update metadata for output files
    dem_meta.update(dtype=rasterio.float32, nodata=np.nan)

    # Define output file paths
    dem_fw = fdem_file.replace('F.tif', '_Fw.tif')
    dem_mw = mask_file.replace('M.tif', '_Mw.tif')

    # Write the final DEM to disk
    with rasterio.open(dem_fw, 'w', **dem_meta) as dest:
        dest.write(final_dem.astype(rasterio.float32), 1)

    # Write the final mask to disk (optional, if you want to save the combined mask)
    with rasterio.open(dem_mw, 'w', **dem_meta) as dest:
        dest.write(combined_mask.astype(rasterio.uint8), 1)

    print(f"Final DEM written to {dem_fw}")
    print(f"Final mask written to {dem_mw}")


def combine_water_masks(lcm_lwm_fn, esa_lwm_fn, wbm_lwm_fn):
    """Combine water masks from multiple sources into a single mask."""

    # Open the LCM mask file
    with rasterio.open(lcm_lwm_fn) as lcm_src:
        lcm_data = lcm_src.read(1)
        mask_meta = lcm_src.meta  # Get metadata for writing output

    # Open the ESA mask file
    with rasterio.open(esa_lwm_fn) as esa_src:
        esa_data = esa_src.read(1)

    # Open the WBM mask file
    with rasterio.open(wbm_lwm_fn) as wbm_src:
        wbm_data = wbm_src.read(1)

    # Combine all masks: keep values where all masks are 0
    combined_mask = (lcm_data == 0) & (esa_data == 0) & (wbm_data == 0)

    # Prepare metadata for output file
    mask_meta.update(dtype=rasterio.uint8, nodata=0)

    # Define output file path
    tname = os.path.basename(lcm_lwm_fn).split('_')[0]
    combined_mask_file = os.path.join(os.path.dirname(lcm_lwm_fn), f'{tname}_LWM.tif')

    # Write the combined mask to disk
    with rasterio.open(combined_mask_file, 'w', **mask_meta) as dest:
        dest.write(combined_mask.astype(rasterio.uint8), 1)

    print(f"Combined water mask written to {combined_mask_file}")

    return combined_mask_file
