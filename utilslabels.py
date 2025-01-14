# if tiles in RNG using pdem else use ldem
# read that paper again [*]

import os 
import glob
import subprocess
import numpy as np
from osgeo import gdal
from concurrent.futures import ProcessPoolExecutor

cpus = int(os.cpu_count() - 4)

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


def gen_labels_by_landcover():
    pass 


def gen_labels_by_otsu():
    pass 


def gen_labels_by_funcion_x():
    pass 

def gen_labels_by_unsupervised():
    # kmeans and others
    pass 

def get_label_name(dsm_file,name):
    dirn = os.path.dirname(dsm_file)
    tilen = basename(dsm_file).split('_')[0] #str(dsm_file).split('\\')[-2]
    return  os.path.join(dirn, f'{tilen}_label_thresh_{name}.tif')


def gen_label_workflow(dsm_path,pdem_path,ldem_path,pmask_path,lmask_path):
    gen_label_by_threshold(dsm_path, pdem_path, pmask_path, threshold=0.5)
    gen_label_by_threshold(dsm_path, ldem_path, lmask_path, threshold=0.5)

#from upaths_wx import labels_dtm_path, labels_dsm_path, labels_dtm_pathB, labels_dsm_pathB
from os.path import basename
from upaths_wx import labels_ldem_pathB, labels_dsm_pathB, labels_pdem_pathB,labels_dsm_path

ldsm_path = labels_dsm_pathB
ldem_path = labels_ldem_pathB
pdem_path = labels_pdem_pathB

if __name__ == '__main__':
    dsm_files = sorted(glob.glob(ldsm_path)); print(len(dsm_files))
    ldem_files = sorted(glob.glob(ldem_path)); print(len(ldem_files))
    pdem_files = sorted(glob.glob(pdem_path)); print(len(pdem_files))
    
    with ProcessPoolExecutor(cpus) as PEX:
        for i in range(len(dsm_files)):
            ldem_file = ldem_files[i]
            pdem_file = pdem_files[i]
            dsm_file = dsm_files[i]
            label_ldem = get_label_name(dsm_files[i],'ldem')
            label_pdem = get_label_name(dsm_files[i],'pdem')
            PEX.submit(
                 gen_label_workflow,dsm_file,pdem_file,
                 ldem_file,label_pdem,label_ldem
            )





            
            #print(label_file)
            #PEX.submit(gen_label_by_threshold,dsm_file, dtm_file, label_file,0.5)
        

# this can be joined with the other pipeline c
#gen_label_by_threshold(dsm_path, dtm_path, mask_path, threshold=0.5)

