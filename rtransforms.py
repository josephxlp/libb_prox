import os
import subprocess
import numpy as np
import rasterio
from sklearn.preprocessing import MinMaxScaler
from osgeo import gdal


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
        print(f"Scaled TIFF already exists at {output_fpath}")
    
    return output_fpath


def dem_derivative(fi, fo, mode):
    valid_modes = ['hillshade', 'slope', 'aspect', 'TRI', 'TPI', 'roughness']
    
    if mode not in valid_modes:
        print(f"Invalid mode. Please choose one of: {', '.join(valid_modes)}")
        return
    
    subprocess.run(['gdaldem', mode, fi, fo])


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
    
    # Create the mask array based on the threshold criteria
    mask = np.where(
        (np.isnan(dsm_data)) | (np.isnan(dtm_data)) | ((dsm_data - dtm_data) < threshold), 1, 0
    )
    
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
    dsm_ds, dtm_ds, mask_ds = None, None, None
    
    print(f"Mask created and saved to {mask_path}")

def roi_gen_label_by_threshold(dsm_path, dtm_path, mask_path, threshold=0.5):
    # Open the DSM and DTM files using rasterio
    with rasterio.open(dsm_path) as dsm_ds, rasterio.open(dtm_path) as dtm_ds:
        if dsm_ds is None or dtm_ds is None:
            raise FileNotFoundError("DSM or DTM file not found.")
        
        # Read the first band (assuming single-band rasters)
        dsm_data = dsm_ds.read(1)
        dtm_data = dtm_ds.read(1)
        
        # Get NoData values
        dsm_nodata = dsm_ds.nodata
        dtm_nodata = dtm_ds.nodata
        
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
        
        # Create the mask array based on the threshold criteria
        mask = np.where(
            (np.isnan(dsm_data)) | (np.isnan(dtm_data)) | ((dsm_data - dtm_data) < threshold), 1, 0
        )
        
        # Create the output mask file using rasterio
        with rasterio.open(mask_path, 'w', driver='GTiff', count=1, dtype='uint8', 
                          width=dsm_ds.width, height=dsm_ds.height, crs=dsm_ds.crs, 
                          transform=dsm_ds.transform) as mask_ds:
            # Write the mask array to the output file
            mask_ds.write(mask, 1)
    
    print(f"Mask created and saved to {mask_path}")



