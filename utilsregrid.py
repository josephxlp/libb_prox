import os 
import subprocess
from osgeo import gdal, gdalconst
import rasterio 
from rasterio import features
import numpy as np
import pandas as pd

from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from upaths import s2bandnames
from utilssentinelii import get_indices, split_raster_by_bands

import utilsvrt as uops 
mem_drv = gdal.GetDriverByName('MEM')
gtif_drv = gdal.GetDriverByName('GTiff')
vrt_drv = gdal.GetDriverByName("VRT")
# do this differently for efficiently 




def scale_tif(fpath):
    output_fpath = fpath.replace('.tif', '__scaled.tif')
    if not os.path.isfile(output_fpath):
        scaler = MinMaxScaler()
        with rasterio.open(fpath) as src:
            data = src.read()
            meta = src.meta
        
        num_bands = data.shape[0]
        scaled_data = np.zeros_like(data, dtype=np.float32)
        for i in range(num_bands):
            band = data[i].reshape(-1, 1)  # Reshape for MinMaxScaler
            scaled_band = scaler.fit_transform(band).reshape(data[i].shape)
            scaled_data[i] = scaled_band


        meta.update(dtype='float32')
        with rasterio.open(output_fpath, 'w', **meta) as dst:
            dst.write(scaled_data)

        print(f"Scaled TIFF saved as {output_fpath}")
    else:
        print(f"Scaled TIFF saved at {output_fpath}")
    return output_fpath

def classify_lwm_TanDEMX_LCM(lcm_fn,lwm_a_fn,lwm_b_fn):
    with rasterio.open(lcm_fn) as src:
        b1 = src.read(1)
        b2 = src.read(2)
        lwm_b1 = np.where(b1 ==3, 1, 0)
        lwb_b2 = np.where((b2 == 1) | (b2 == 2), 1, 0)
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1)
        #lwm_a_fn = lcm_fn.replace('.tif', '_LWM_A.tif')
        with rasterio.open(lwm_a_fn, 'w', **meta) as dst:
            dst.write(lwm_b1, 1)

        #lwm_b_fn = lcm_fn.replace('.tif', '_LWM_B.tif')
        with rasterio.open(lwm_b_fn, 'w', **meta) as dst:
            dst.write(lwb_b2, 1)
    
    return lwm_a_fn, lwm_b_fn

def classify_lwm_ESAWC(fipath, fopath,water_code=80):
    with rasterio.open(fipath) as src:
        data = src.read(1)
        lwm = (data == water_code).astype(np.uint8)
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(fopath, 'w', **meta) as dst:
            dst.write(lwm, 1)


def classify_lwm_CopWBM(fipath, fopath):
    # water_classes = {1, 2, 3}  # Ocean, Lake, River
    water_classes = {1, 2, 3}
    with rasterio.open(fipath) as src:
        data = src.read(1)
        lwm = np.isin(data, list(water_classes)).astype(np.uint8)
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(fopath, 'w', **meta) as dst:
            dst.write(lwm, 1)


def classify_lwm_TanDEMX_WAM(fipath, fopath, lthreshold=33, htheshold=127):
    # 3 to 127 or 33 to 127, max, and more accurate resectively for lthresh
    """
    Classifies pixels in the TanDEM-X WAM into 'water' and 'non-water'
    and saves the result as a binary raster.

    Parameters:
    - input_path: str, path to the input TanDEM-X WAM dataset (GeoTIFF).
    - output_path: str, path to save the output binary raster (GeoTIFF).
    - threshold_min: int, minimum threshold value for water detection (default is 3).
    - threshold_max: int, maximum threshold value for water detection (default is 127).
    """ 
    with rasterio.open(fipath) as src:
        data = src.read(1)
        lwm = ((data >= lthreshold) & (data <= htheshold)).astype(np.uint8)
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(fopath, 'w', **meta) as dst:
            dst.write(lwm, 1)


def dem_derivative(fi, fo, mode):
    if mode == 'hillshade':
        subprocess.run(['gdaldem', 'hillshade', fi, fo])
    elif mode == 'slope':
        subprocess.run(['gdaldem', 'slope', fi, fo])
    elif mode == 'aspect':
        subprocess.run(['gdaldem', 'aspect', fi, fo])
    elif mode == 'TRI':
        subprocess.run(['gdaldem', 'TRI', fi, fo])
    elif mode == 'TPI':
        subprocess.run(['gdaldem', 'TPI', fi, fo])
    elif mode == 'roughness':
        subprocess.run(['gdaldem', 'roughness', fi, fo])
    else:
        print("Invalid mode. Please choose one of: hillshade, slope, aspect, TRI, TPI, roughness")


def gen_label_by_threshold(dsm_path, dtm_path, mask_path, threshold=0.5):
    # Open the DSM and DTM files
    dsm_ds = gdal.Open(dsm_path)
    dtm_ds = gdal.Open(dtm_path)
    
    if dsm_ds is None or dtm_ds is None:
        raise FileNotFoundError("DSM or DTM file not found.")
    
    # Read the first band (assuming single-band rasters)
    dsm_band = dsm_ds.GetRasterBand(1)
    dtm_band = dtm_ds.GetRasterBand(1)
    
    dsm_data = dsm_band.ReadAsArray()
    dtm_data = dtm_band.ReadAsArray()
    
    # Get NoData values from the bands
    dsm_nodata = dsm_band.GetNoDataValue()
    dtm_nodata = dtm_band.GetNoDataValue()
    
    # Set NoData values to np.nan
    if dsm_nodata is not None:
        dsm_data[dsm_data == dsm_nodata] = np.nan
    if dtm_nodata is not None:
        dtm_data[dtm_data == dtm_nodata] = np.nan
    
    # Filter values < -999 and > 1000 and set to np.nan
    dsm_data[(dsm_data < -999) | (dsm_data > 1000)] = np.nan
    dtm_data[(dtm_data < -999) | (dtm_data > 1000)] = np.nan
    
    # Ensure both arrays have the same shape
    if dsm_data.shape != dtm_data.shape:
        raise ValueError("DSM and DTM must have the same dimensions.")
    
    # Create the mask array based on the criteria
    mask = np.where((np.isnan(dsm_data)) | (np.isnan(dtm_data)) | ((dsm_data - dtm_data) < threshold), 1, 0)
    
    # Create the output mask file
    driver = gdal.GetDriverByName('GTiff')
    mask_ds = driver.Create(mask_path, dsm_ds.RasterXSize, dsm_ds.RasterYSize, 1, gdal.GDT_Byte)
    
    # Set the same georeference as the input files
    mask_ds.SetGeoTransform(dsm_ds.GetGeoTransform())
    mask_ds.SetProjection(dsm_ds.GetProjection())
    
    # Write the mask array to the output file
    mask_band = mask_ds.GetRasterBand(1)
    mask_band.WriteArray(mask)
    
    # Close the datasets
    dsm_ds = None
    dtm_ds = None
    mask_ds = None
    
    print(f"Mask created and saved to {mask_path}")


def format_tile_fpath(tile_dpath, tilename, file_fpath):
    tile_fpath = os.path.join(tile_dpath, 
                              tilename + '_' + os.path.basename(file_fpath))
    tile_fpath = vrt_extentison_to_tif(tile_fpath)
    return tile_fpath

def vrt_extentison_to_tif(path):
    return path.replace('.vrt', '.tif')

def get_nodata_value(raster_path):
    with rasterio.open(raster_path) as src:
        nodata_value = src.nodata
    return nodata_value


def cat_regrid_params():
    ndv = 0
    algo = 'near'
    dtype = 'Byte'
    return ndv, algo, dtype


def num_regrid_params():
    ndv = -9999.
    dtype = 'Float32'
    algo = 'bilinear'
    return ndv, algo, dtype 

def gdal_regrid(fi, fo, xmin, ymin, xmax, ymax, xres, yres,
                mode, t_epsg='EPSG:4979', overwrite=False):
    if mode == 'num':
        ndv, algo, datatype = num_regrid_params()
    elif mode == 'cat':
        ndv, algo, datatype = cat_regrid_params()

    dst_ndv = ndv 
    src_ndv = get_nodata_value(fi)
    print('src ndv:', src_ndv)
    print('dst ndv:', dst_ndv)
    try:
        src_nodata_str = f"-srcnodata {src_ndv}" if src_ndv is not None else ""
        dst_nodata_str = f"-dstnodata {dst_ndv}" if dst_ndv is not None else ""
        
        overwrite_option = "-overwrite" if overwrite else ""

        # Calculate output width and height based on input resolution
        output_width = round((xmax - xmin) / xres)
        output_height = round((ymax - ymin) / abs(yres))

        cmd = (f'gdalwarp -ot {datatype} -multi {overwrite_option} '
               f'-te {xmin} {ymin} {xmax} {ymax} '
               f'-ts {output_width} {output_height} '
               f'-r {algo}  -t_srs {t_epsg} -tr {xres} {yres} -tap '
               f'-co compress=lzw -co num_threads=all_cpus '
               f'-co TILED=YES '
               f'{src_nodata_str} {dst_nodata_str} '
               f'{fi} {fo}')

        os.system(cmd)
    except Exception as e:
        print(f"Error: {e}")


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

def get_band_ndv(band):
    """Determine the NoData value for a raster band."""
    ndv = band.GetNoDataValue()
    if ndv is None:
        data_array = band.ReadAsArray()
        max_val = np.max(data_array)
        min_val = np.min(data_array)

        if min_val < 0 and min_val <= -9999:
            ndv = min_val
        elif max_val > abs(-9999.):
            ndv = max_val
    return ndv

def band_get_masked_array(band):
    """Return a masked array of the raster band."""
    ndv = get_band_ndv(band)
    return np.ma.masked_values(band.ReadAsArray(), ndv)

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

    if os.path.exists(output_file):
        os.remove(output_file)

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        output_file, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(ds.GetGeoTransform())
    dataset.SetProjection(ds.GetProjection())
    band = dataset.GetRasterBand(1)
    band.WriteArray(mask.astype(np.uint8))
    dataset = None

def write_geotif(input_file, output_file, data_array, ndvalue=-9999.):
    """Write a masked array to a new GeoTIFF file."""
    ds = open_ds(input_file)
    if ds is None:
        return

    if os.path.exists(output_file):
        os.remove(output_file)

    data_array.set_fill_value(ndvalue)
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        output_file, ds.RasterXSize, ds.RasterYSize,
        ds.RasterCount, ds.GetRasterBand(1).DataType)
    dataset.SetMetadata(ds.GetMetadata())
    dataset.SetGeoTransform(ds.GetGeoTransform())
    dataset.SetProjection(ds.GetProjection())
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(ndvalue)
    band.WriteArray(data_array.filled())
    dataset = None

def filter_tandemx_noise(dem_file, hem_file, com_file, n_iter=1):
    fdem_file = dem_file.replace('.tif', '_F.tif')
    mask_file = dem_file.replace('.tif', '_M.tif')

    """Process DEM data by applying masks and write the result to a new file."""
    gdal.UseExceptions()
    print('Loading and preprocessing DEM')
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

    max_err_multi = 1.5
    mask = np.logical_or(mask, (hem.data > max_err_multi))
    com_invalid = (0, 1, 2)
    mask = np.logical_or(mask, np.isin(com.data, com_invalid))

    print('Applying Masks')
    dem_masked = np.ma.array(dem, mask=mask)
    mask = ndimage.binary_dilation(mask, iterations=n_iter)
    mask = ndimage.binary_erosion(mask, iterations=n_iter)

    # Write the mask to a new GeoTIFF file
    write_mask_as_geotif(dem_file, mask, mask_file)

    dem_masked = np.ma.array(dem, mask=mask)
    ndvalue = get_band_ndv(band)
    write_geotif(dem_file, fdem_file, dem_masked, ndvalue)
    return fdem_file, mask_file

def filter_water(fdem_file, mask_file, lcm_lwm_fn, esa_lwm_fn, wbm_lwm_fn):
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

    # Prepare metadata for output files
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


def regrid_datasets(

    ds_tiles_dpath, tilename, xmin, ymin, xmax, ymax, xres, yres,
    tdem_dem_fpath, tdem_hem_fpath, tdem_wam_fpath, tdem_com_fpath, 
    cdem_wbm_fpath, esawc_fpath, dtm_fpath, 
    pdem_fpath, cdem_dem_fpath, edem_dem_fpath,
    egm08_fpath,edem_edem_W84_fpath,egm96_fpath,edem_lcm_fpath,
    s1_fpath, s2_fpath
    ):

    tilename_dpath = os.path.join(ds_tiles_dpath, tilename)
    os.makedirs(tilename_dpath,exist_ok=True)
    ds = {}
    s1_tile = format_tile_fpath(tilename_dpath, tilename, s1_fpath)
    gdal_regrid(s1_fpath, s1_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['s1'] = s1_tile
    

    s2_tile = format_tile_fpath(tilename_dpath, tilename, s2_fpath)
    gdal_regrid(s2_fpath, s2_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['s2'] = s2_tile

    bandfiles = split_raster_by_bands(s2_tile,s2bandnames)
    bfiles = bandfiles[:5]
    afiles = bandfiles[5:]
    for afile in afiles: 
        print(f'deleing in split_raster_by_bands {afile}')
        os.remove(afile)
        get_indices(bfiles)

    
    # clipping 
    tdem_dem_tile = format_tile_fpath(tilename_dpath, tilename, tdem_dem_fpath)
    gdal_regrid(tdem_dem_fpath, tdem_dem_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['tdem_dem'] = tdem_dem_tile

    tdem_hem_tile = format_tile_fpath(tilename_dpath, tilename, tdem_hem_fpath)
    gdal_regrid(tdem_hem_fpath, tdem_hem_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['tdem_hem'] = tdem_hem_tile
    
    #[] hem mask 

    tdem_wam_tile = format_tile_fpath(tilename_dpath, tilename, tdem_wam_fpath)
    gdal_regrid(tdem_wam_fpath, tdem_wam_tile, xmin, ymin, xmax, ymax, xres, yres, mode='cat')
    ds['tdem_wam'] = tdem_wam_tile

    tdem_com_tile = format_tile_fpath(tilename_dpath, tilename, tdem_com_fpath)
    gdal_regrid(tdem_com_fpath, tdem_com_tile, xmin, ymin, xmax, ymax, xres, yres, mode='cat')
    ds['tdem_com'] = tdem_com_tile
    #[] com mask 

    cdem_wbm_tile = format_tile_fpath(tilename_dpath, tilename, cdem_wbm_fpath)
    gdal_regrid(cdem_wbm_fpath, cdem_wbm_tile, xmin, ymin, xmax, ymax, xres, yres, mode='cat')
    ds['cdem_wbm'] = cdem_wbm_tile

    edem_lcm_tile = format_tile_fpath(tilename_dpath, tilename, edem_lcm_fpath)
    gdal_regrid(edem_lcm_fpath, edem_lcm_tile, xmin, ymin, xmax, ymax, xres, yres, mode='cat')
    ds['edem_lcm'] = edem_lcm_tile

    
    esawc_tile = format_tile_fpath(tilename_dpath, tilename, esawc_fpath)
    gdal_regrid(esawc_fpath, esawc_tile, xmin, ymin, xmax, ymax, xres, yres, mode='cat')
    ds['esawc'] = esawc_tile

    # add if already exist 
    wbm_lwm_fn = cdem_wbm_tile.replace('cdem_WBM.tif', 'cdem_WBM_LWM.tif')
    if not os.path.isfile(wbm_lwm_fn):
        classify_lwm_CopWBM(cdem_wbm_tile, wbm_lwm_fn)
    ds['cdem_wbm'] = wbm_lwm_fn

    esa_lwm_fn = esawc_tile.replace('_multi_ESAWC.tif', '_ESAWC_LWM.tif')
    if not os.path.isfile(esa_lwm_fn):
        classify_lwm_ESAWC(esawc_tile, esa_lwm_fn,water_code=80)
    ds['esawc_lwm'] = esa_lwm_fn

 
    lwm_a_fn = edem_lcm_tile.replace('.tif', '_LWM_A.tif')
    lwm_b_fn = edem_lcm_tile.replace('.tif', '_LWM_B.tif')
    if not os.path.isfile(lwm_a_fn):
        classify_lwm_TanDEMX_LCM(edem_lcm_tile,lwm_a_fn,lwm_b_fn)
        #elcm_fna, elcm_fnb = classify_lwm_TanDEMX_LCM(edem_lcm_tile,lwm_a_fn,lwm_b_fn)


    ds['edem_lcm_lwma'] = lwm_a_fn
    ds['edem_lcm_lwmb'] = lwm_b_fn
    lcm_lwm_fn = lwm_a_fn

    fdem_file = tdem_dem_tile.replace('.tif', '_F.tif')
    mask_file = tdem_dem_tile.replace('.tif', '_M.tif')
    if not os.path.isfile(fdem_file):
    
        fdem_file, mask_file = filter_tandemx_noise(tdem_dem_tile, 
                                                    tdem_hem_tile, 
                                                    tdem_com_tile, 
                                                    n_iter=1)
    
    ds['tdem_dem_f'] = fdem_file
    ds['tdem_dem_m'] = mask_file
    
    
    dem_fw = fdem_file.replace('F.tif', '_Fw.tif')
    dem_mw = mask_file.replace('M.tif', '_Mw.tif')
    if not os.path.isfile(dem_fw):
        filter_water(fdem_file, mask_file, lcm_lwm_fn, esa_lwm_fn, wbm_lwm_fn)

    ds['tdem_dem_fw'] = dem_fw
    ds['tdem_dem_mw'] = dem_mw

    tname = os.path.basename(lcm_lwm_fn).split('_')[0]
    combined_mask_file = os.path.join(os.path.dirname(lcm_lwm_fn), f'{tname}_LWM.tif')
    if not os.path.isfile(combined_mask_file):
        combined_mask_file = combine_water_masks(lcm_lwm_fn, esa_lwm_fn, wbm_lwm_fn)
    ds['lwm'] = combined_mask_file
 
    #ethchm_tile = format_tile_fpath(tilename_dpath, tilename, ethchm_fpath)
    #gdal_regrid(ethchm_fpath, ethchm_tile, xmin, ymin, xmax, ymax, xres, yres, mode='cat')
    #ds['eth'] = ethchm_tile

    #gfc_tile = format_tile_fpath(tilename_dpath, tilename, gfc_fpath)
    #gdal_regrid(gfc_fpath, gfc_tile, xmin, ymin, xmax, ymax, xres, yres, mode='cat')
    #ds['gfc'] = ethchm_tile

    #wsf2d_tile = format_tile_fpath(tilename_dpath, tilename, wsf2d_file)
    #gdal_regrid(wsf2d_file, wsf2d_tile, xmin, ymin, xmax, ymax, xres, yres, mode='cat')
    #ds['wsf2d'] = wsf2d_tile

    ldar_tile = format_tile_fpath(tilename_dpath, tilename, dtm_fpath)
    gdal_regrid(dtm_fpath, ldar_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['ldtm'] = ldar_tile

    # dsm_tile = format_tile_fpath(tilename_dpath, tilename, dsm_fpath)
    # gdal_regrid(dsm_fpath, dsm_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    # ds['ldsm'] = dsm_tile

    pdem_tile = format_tile_fpath(tilename_dpath, tilename, pdem_fpath)
    gdal_regrid(pdem_fpath, pdem_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['pdem'] = pdem_tile

    cdem_dem_tile = format_tile_fpath(tilename_dpath, tilename, cdem_dem_fpath)
    gdal_regrid(cdem_dem_fpath, cdem_dem_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['cdem_dem'] = cdem_dem_tile

    edem_dem_tile = format_tile_fpath(tilename_dpath, tilename, edem_dem_fpath)
    gdal_regrid(edem_dem_fpath, edem_dem_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['edem_dem'] = edem_dem_tile

    edem_demw84_tile = format_tile_fpath(tilename_dpath, tilename, edem_edem_W84_fpath)
    gdal_regrid(edem_edem_W84_fpath, edem_demw84_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['edem_demw84'] = edem_demw84_tile 

    #cdem_demw84_tile
    #ds['cdem_demw84'] = cdem_dem_tile

    # wsfba_tile = format_tile_fpath(tilename_dpath, tilename, wsfba_fpath)
    # gdal_regrid(wsfba_fpath, wsfba_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    # ds['wsfba'] = wsfba_tile

    # wsfbf_tile = format_tile_fpath(tilename_dpath, tilename, wsfbf_fpath)
    # gdal_regrid(wsfbf_fpath, wsfbf_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    # ds['wsfbf'] = wsfbf_tile

    # wsfbh_tile = format_tile_fpath(tilename_dpath, tilename, wsfbh_fpath)
    # gdal_regrid(wsfbh_fpath, wsfbh_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    # ds['wsfbh'] = wsfbh_tile

    # wsfbv_tile = format_tile_fpath(tilename_dpath, tilename, wsfbv_fpath)
    # gdal_regrid(wsfbv_fpath, wsfbv_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    # ds['wsfbv'] = wsfbv_tile

    egm08_tile = format_tile_fpath(tilename_dpath, tilename, egm08_fpath)
    gdal_regrid(egm08_fpath, egm08_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['egm08'] = egm08_tile
   

    egm96_tile = format_tile_fpath(tilename_dpath, tilename, egm96_fpath)
    gdal_regrid(egm96_fpath, egm96_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['egm96'] = egm96_tile



    s2_tilex = scale_tif(s2_tile)
    ds['s2x'] = s2_tilex

    s1_tilex = scale_tif(s1_tile)
    ds['s1x'] = s1_tilex

    egm96_tilex = scale_tif(egm96_tile)
    ds['egm96x'] = egm96_tilex

    egm08_tilex = scale_tif(egm08_tile)
    ds['egm08x'] = egm08_tilex

    #tdem_dem_clean_tile = tdem_dem_tile.replace('.tif', '_clean.tif')
    #tdem_erode_tile = tdem_dem_tile.replace('.tif', '_E.tif')

    # fill the values with edem []

    # dem derivatives 

    # #if not os.path.isfile(tdem_dem_clean_tile):
    # if not os.path.isfile(tdem_erode_tile):
    #     remove_tandemx_noise(
    #     tdem_dem_tile, cdem_wbm_tile, tdem_hem_tile, tdem_com_tile, 
    #     tdem_erode_tile, n_iter=1, ndvalue=-9999.)

    # #ds['tdem_dem_clean'] = tdem_dem_clean_tile
    # ds['tdem_dem_e'] = tdem_erode_tile
    # tdem_ef_tile = tdem_dem_tile.replace('.tif', '_EF.tif')
    # lwm_tile = cdem_wbm_tile.replace('_cdem_WBM.tif','_LWM.tif')

    # if not os.path.isfile(tdem_ef_tile):
    #     process_dem_and_water_mask(dem_file=tdem_erode_tile, 
    #                             water_mask_file=cdem_wbm_tile, 
    #                             filtered_dem_file=tdem_ef_tile, 
    #                             land_water_mask_file=lwm_tile)
    # ds['tdem_dem_ef'] = tdem_ef_tile
    # ds['tdem_lwm'] = lwm_tile

    # tdem_dem_clean_tile_bmask = tdem_dem_clean_tile.replace('.tif', '_binmask.tif')
    # lthresh, hthresh = -99, 1000
    # if not os.path.isfile(tdem_dem_clean_tile_bmask):
    #     process_binmask(tdem_dem_clean_tile, tdem_dem_clean_tile_bmask, lthresh, hthresh)

    # ds['tdem_dem_clean_binmask'] = tdem_dem_clean_tile_bmask

    #tdem_dem_clean_tile_filled
    # tdem_filled_tile = tdem_ef_tile.replace('.tif', '_filled.tif')
    # if not os.path.isfile(tdem_filled_tile):
    #     # tdem_erode_tile ::tdem_ef_tile
    #     fill_nodata(src_file=tdem_erode_tile, dst_file=tdem_filled_tile, 
    #                 max_distance=100,smoothing_iterations=2)
    # ds['tdem_dem_filled'] = tdem_filled_tile

    # ldem_label_tile = tdem_dem_tile.replace('.tif', '_label_LdemTHRESH.tif')
    # if not os.path.isfile(ldem_label_tile):
    #     gen_label_by_threshold(dsm_path=edem_demw84_tile, dtm_path=ldar_tile, mask_path=ldem_label_tile, threshold=0.5)
    # ds['ldem_label'] = ldem_label_tile

    # pdem_label_tile = tdem_dem_tile.replace('.tif', '_label_PdemTHRESH.tif')
    # if not os.path.isfile(ldem_label_tile):
    #     gen_label_by_threshold(dsm_path=edem_demw84_tile, dtm_path=pdem_tile, mask_path=pdem_label_tile, threshold=0.5)
    # ds['pdem_label'] = pdem_label_tile

        
    # tile_slp = tdem_filled_tile.replace('.tif', '_slp.tif')
    # if not os.path.isfile(tile_slp):
    #     dem_derivative(tdem_filled_tile, tile_slp, 'slope')
    #     run_gdal_fillnodata(tile_slp,tile_slp)
    # ds['tdem_dem_slp'] = tile_slp

    # tile_hsd = tdem_filled_tile.replace('.tif', '_hsd.tif')
    # if not os.path.isfile(tile_hsd):
    #     dem_derivative(tdem_filled_tile, tile_hsd, 'hillshade')
    #     run_gdal_fillnodata(tile_hsd,tile_hsd)
    # ds['tdem_dem_hsd'] = tile_hsd

    # tile_tri = tdem_filled_tile.replace('.tif', '_tri.tif')
    # if not os.path.isfile(tile_tri):
    #     dem_derivative(tdem_filled_tile, tile_tri, 'TRI')
    #     run_gdal_fillnodata(tile_tri,tile_tri)
    # ds['tdem_dem_tri'] = tile_tri

    # tile_tpi = tdem_filled_tile.replace('.tif', '_tpi.tif')
    # if not os.path.isfile(tile_tpi):
    #     dem_derivative(tdem_filled_tile, tile_tpi, 'TPI')
    #     run_gdal_fillnodata(tile_tpi,tile_tpi)
    # ds['tdem_dem_tpi'] = tile_tpi

    # tile_rgx = tdem_filled_tile.replace('.tif', '_rgx.tif')
    # if not os.path.isfile(tile_rgx):
    #     dem_derivative(tdem_filled_tile, tile_rgx, 'roughness')
    #     run_gdal_fillnodata(tile_rgx,tile_rgx)
    # ds['tdem_dem_rgx'] = tile_rgx

    yaml_tile = os.path.join(tilename_dpath, f'{tilename}_ds.yaml')
    # csv_tile = os.path.join(tilename_dpath, f'{tilename}_ds.csv')
    # dsf = pd.DataFrame(ds)
    # dsf.to_csv(csv_tile, index=False)
    uops.write_yaml(ds, yaml_tile)
    print('yaml_tile:', yaml_tile)
    print('Already exists!!')
    print(ds)

def get_raster_info(tif_path):
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    xres = geotrans[1]
    yres = geotrans[5]
    w = ds.RasterXSize
    h = ds.RasterYSize
    # Get bounding box of input dataset
    xmin = geotrans[0]
    ymax = geotrans[3]
    xmax = xmin + (xres * w)  # Fix calculation for xmax
    ymin = ymax + (yres * h)  # Fix calculation for ymin
    ds = None
    return proj, xres, yres, xmin, xmax, ymin, ymax, w, h


def process_tile(
        basefile, ds_tiles_dpath, tdem_dem_fpath, tdem_hem_fpath, 
        tdem_wam_fpath, tdem_com_fpath, cdem_wbm_fpath, esawc_fpath, 
        dtm_fpath, pdem_fpath, cdem_dem_fpath, 
        edem_dem_fpath,egm08_fpath,edem_edem_W84_fpath,egm96_fpath,
        edem_lcm_fpath,s1_fpath, s2_fpath):
    
    tilename = uops.get_tilename_from_tdem_basename(basefile)
    print(os.path.basename(basefile))
    print(tilename)
    tilename_dpath = os.path.join(ds_tiles_dpath, tilename)
    os.makedirs(tilename_dpath,exist_ok=True)
    tile_fpath = format_tile_fpath(tilename_dpath, tilename, tdem_dem_fpath) 
    proj, xres, yres, xmin, xmax, ymin, ymax, w, h = get_raster_info(basefile)
    print('dst size:', w, h)
    regrid_datasets(

    ds_tiles_dpath, tilename, xmin, ymin, xmax, ymax, xres, yres,
    tdem_dem_fpath, tdem_hem_fpath, tdem_wam_fpath, tdem_com_fpath, 
    cdem_wbm_fpath, esawc_fpath, dtm_fpath,
    pdem_fpath, cdem_dem_fpath, edem_dem_fpath,egm08_fpath,edem_edem_W84_fpath,
    egm96_fpath,edem_lcm_fpath,s1_fpath, s2_fpath
    )

    