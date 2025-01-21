import os
import rasterio
from osgeo import gdal, gdalconst  # all gdal function to rasterio 

def gdal_regrid(fi, fo, xmin, ymin, xmax, ymax, xres, yres,
                mode, t_epsg='EPSG:4979', overwrite=False):
    # Determine regrid parameters based on mode
    if mode == 'num':
        ndv, algo, datatype = num_regrid_params()
    elif mode == 'cat':
        ndv, algo, datatype = cat_regrid_params()
    
    dst_ndv = ndv
    src_ndv = get_nodata_value(fi)
    
    print(f"Source NoData Value: {src_ndv}")
    print(f"Destination NoData Value: {dst_ndv}")

    try:
        # Set NoData values for source and destination
        src_nodata_str = f"-srcnodata {src_ndv}" if src_ndv is not None else ""
        dst_nodata_str = f"-dstnodata {dst_ndv}" if dst_ndv is not None else ""
        
        # Set overwrite option if required
        overwrite_option = "-overwrite" if overwrite else ""

        # Calculate output width and height based on input resolution
        output_width = round((xmax - xmin) / xres)
        output_height = round((ymax - ymin) / abs(yres))

        # Construct the GDAL warp command
        cmd = (f'gdalwarp -ot {datatype} -multi {overwrite_option} '
               f'-te {xmin} {ymin} {xmax} {ymax} '
               f'-ts {output_width} {output_height} '
               f'-r {algo} -t_srs {t_epsg} -tr {xres} {yres} -tap '
               f'-co compress=lzw -co num_threads=all_cpus '
               f'-co TILED=YES '
               f'{src_nodata_str} {dst_nodata_str} '
               f'{fi} {fo}')

        # Execute the command
        os.system(cmd)
    
    except Exception as e:
        print(f"Error: {e}")

def cat_regrid_params():
    """Return parameters for categorical regridding."""
    ndv = 0
    algo = 'near'
    dtype = 'Byte'
    return ndv, algo, dtype


def num_regrid_params():
    """Return parameters for numerical regridding."""
    ndv = -9999.0
    algo = 'bilinear'
    dtype = 'Float32'
    return ndv, algo, dtype

def vrt_extension_to_tif(path):
    """Convert a .vrt file path to a .tif file path."""
    return path.replace('.vrt', '.tif')

def format_tile_fpath(tile_dpath, tilename, file_fpath):
    """Generate a formatted file path for a tile based on input parameters."""
    print(file_fpath)
    tile_fpath = os.path.join(tile_dpath, f"{tilename}_{os.path.basename(file_fpath)}")
    tile_fpath = vrt_extension_to_tif(tile_fpath)
    return tile_fpath


def get_nodata_value(raster_path):
    """Get the NoData value from a raster file."""
    with rasterio.open(raster_path) as src:
        nodata_value = src.nodata
    return nodata_value


def get_raster_info(tif_path):
    """Extract and return raster information including projection, resolution, bounding box, and dimensions."""
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    
    # Fetch the projection and geotransform
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    
    # Extract resolution and raster dimensions
    xres = geotrans[1]
    yres = geotrans[5]
    w = ds.RasterXSize
    h = ds.RasterYSize
    
    # Calculate bounding box
    xmin = geotrans[0]
    ymax = geotrans[3]
    xmax = xmin + (xres * w)
    ymin = ymax + (yres * h)  # Note: yres is negative, which ensures ymax > ymin for most datasets
    
    # Close the dataset
    ds = None
    
    return proj, xres, yres, xmin, xmax, ymin, ymax, w, h
