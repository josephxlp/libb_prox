import os
import subprocess
from osgeo import gdal
from glob import glob 
from concurrent.futures import ProcessPoolExecutor
from upaths import TILES12_DPATH,TILESX_DPATH,OPEN_TOPOGRAPHY_DPATH


def list_base_files(base_dpath, varname):
    bpattern = f"{base_dpath}/*/*/{varname}/*tif"
    #print(bpattern)
    bfiles = glob(bpattern)
    print(f'{varname} {len(bfiles)} files')
    return bfiles

def filter_files_by_endingwith(files, var_ending):
    filtered_files = [f for f in files if any(f.endswith(ending) for ending in var_ending)]
    print(f"Filtered files count: {len(filtered_files)}/{len(files)}")
    return filtered_files


def resample_raster(input_raster, output_raster, target_resolution, output_format="GTiff"):
    # Check if output raster already exists
    if os.path.exists(output_raster):
        print(f"Output raster {output_raster} already exists. Skipping resampling.")
        return
    
    # Open the raster to check its properties
    raster = gdal.Open(input_raster)
    
    if raster is None:
        raise FileNotFoundError(f"Input raster {input_raster} not found.")
    
    # Determine the data type and whether it's categorical or numerical
    band = raster.GetRasterBand(1)
    data_type = band.DataType
    is_categorical = False

    # For simplicity, consider raster data as categorical if the data type is integer
    if data_type in [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32]:
        is_categorical = True

    # Set resampling method based on data type
    resampling_method = "near" if is_categorical else "bilinear"

    # Prepare gdalwarp command
    target_resolution_x, target_resolution_y = target_resolution
    command = [
        "gdalwarp",
        "-tr", str(target_resolution_x), str(target_resolution_y),
        "-t_srs", "EPSG:4979",
        "-r", resampling_method,
        "-of", output_format,
        input_raster,
        output_raster
    ]

    # Execute the gdalwarp command using subprocess
    try:
        subprocess.run(command, check=True)
        print(f"Resampling completed: {output_raster}")
    except subprocess.CalledProcessError as e:
        print(f"Error in resampling raster: {e}")

def get_raster_resolution(input_raster):
    # Open the raster file
    raster = gdal.Open(input_raster)
    
    if raster is None:
        raise FileNotFoundError(f"Input raster {input_raster} not found.")
    
    # Get the geotransform (affine transformation)
    geotransform = raster.GetGeoTransform()
    
    if geotransform is None:
        raise ValueError(f"Could not get geotransform for raster {input_raster}.")
    
    # The resolution is determined by the pixel size in the x and y directions
    pixel_resolution_x = geotransform[1]  # Pixel size in the x direction
    pixel_resolution_y = -geotransform[5]  # Pixel size in the y direction (negative due to coordinate system)
    
    return pixel_resolution_x, pixel_resolution_y

def match_baseline_to_12m_files(basefile,D12PATH,DXPATH):
    tilename = basefile.split('/')[-3]
    print(tilename)
    t12path = os.path.join(D12PATH, tilename)
    #tXdpath = os.path.join(DXPATH, 'RESAMPLE',tilename)
    tXdpath = os.path.join(DXPATH,tilename)
    os.makedirs(tXdpath, exist_ok=True)
    t12files = glob(f'{t12path}/*.tif')
    t12files = filter_files_by_endingwith(t12files, VAROI)
    txfiles = [os.path.join(tXdpath,os.path.basename(i)) for i in t12files]
    assert len(t12files) == len(txfiles), 'filelist do not match'
    return t12files, txfiles



varname_list = ['COP30', 'COP90','GEBCOSubIceTopo', 'GEDI_L3']
GRIDLIST = [30,90,500,1000]
VAROI = RESAMPLE_VAR_ENGING = [
    'EGM08.tif','EGM96.tif', 'tdem_HEM.tif','S1.tif','S2.tif'
    'NegroAOIDTM.tif', 'multi_DTM_LiDAR.tif','tdem_DEM.tif','edem_W84.tif',
    'tdem_DEM__Fw.tif','cdem_DEM.tif']
RESAMPLE_VAR_SPECTIAL_ENGING = ['multi_ESAWC.tif']#,

def process_raster(varname, GRID, TILESX_DPATH, OPEN_TOPOGRAPHY_DPATH, TILES12_DPATH, DXPATH):
    """
    Processes rasters in parallel for each varname and GRID.
    """
    print(f'{varname}::{DXPATH}')
    basefiles = list_base_files(OPEN_TOPOGRAPHY_DPATH, varname)
    
    for basefile in basefiles:
        resolution = get_raster_resolution(basefile)
        t12files, txfiles = match_baseline_to_12m_files(basefile, TILES12_DPATH, DXPATH)

        for fi, fo in zip(t12files, txfiles):
            resample_raster(fi, fo, resolution)



def main():
    with ProcessPoolExecutor() as executor:
        futures = []
        
        # Iterate over varname_list and GRIDLIST to process them in parallel
        for i, varname in enumerate(varname_list):
            #if i > 0: break  # Uncomment to process more variables if needed
            GRID = GRIDLIST[i]
            DXPATH = f'{TILESX_DPATH}{GRID}'
            os.makedirs(DXPATH, exist_ok=True)
            
            # Submit a task for each varname
            futures.append(executor.submit(process_raster, varname, GRID, TILESX_DPATH, OPEN_TOPOGRAPHY_DPATH, TILES12_DPATH, DXPATH))
        
        # Wait for all futures to complete
        for future in futures:
            future.result()  # This blocks until the task is finished

if __name__ == '__main__':
    main()

# for i, varname in enumerate(varname_list):
#     if i > 0: break
#     GRID = GRIDLIST[i]
#     DXPATH = f'{TILESX_DPATH}{GRID}'
#     os.makedirs(DXPATH, exist_ok=True)  
#     print(f'{varname}::{DXPATH}')
#     basefiles = list_base_files(OPEN_TOPOGRAPHY_DPATH,varname)
#     for j, basefile in enumerate(basefiles):
#         #print(basefile)
#         resolution = get_raster_resolution(basefile)
#         t12files, txfiles = match_baseline_to_12m_files(basefile,TILES12_DPATH,DXPATH)

#         for k in range(len(t12files)):
#             fi = t12files[0]
#             fo = txfiles[0]
#             resample_raster(fi, fo, resolution)