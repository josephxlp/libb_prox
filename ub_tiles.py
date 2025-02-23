import os 
from rgrid import format_tile_fpath, gdal_regrid, get_raster_info
from rfilter import (classify_lwm_CopWBM, classify_lwm_CopWBM, classify_lwm_TanDEMX_LCM,
                     filter_tandemx_noise,filter_water,combine_water_masks,classify_lwm_ESAWC)
#from rlabels import gen_label_by_man_threshold#, generate_adaptive_height_mask
# from rulabels import (generate_adaptive_height_mask,gen_normalised_mask,gen_multiclass_mask, 
#                       gen_binary_landcover_mask,generate_adaptive_slope_landcover_mask,
#                       gen_simplified_landcover_mask,generate_adaptive_slope_landcover_maskP,
#                       gen_local_adaptive_mask,unsupervised_landcover_mask,
#                       dbscan_landcover_mask,kmeans_classify_dsm_dtm)
# from rtransforms import dem_derivative
# from rlabels import (label_kmeans,label_slope_adathresh,label_lcmultithresh,
#                      label_lcmultithresh_slope,label_landdata,label_multiclass,
#                      label_normthresh,label_height_adathresh)
import ua_vrts as uops 
from rtransforms import scale_raster,raster_calc

def retile_datasets(
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
    
    print(s2_fpath)
    s2_tile = format_tile_fpath(tilename_dpath, tilename, s2_fpath)
    gdal_regrid(s2_fpath, s2_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['s2'] = s2_tile

    s2x_tile = scale_raster(s2_tile, method="minmax")
    ds['s2x'] = s2x_tile

    s1x_tile = scale_raster(s1_tile, method="minmax")
    ds['s1x'] = s1x_tile

       # clipping 
    tdem_dem_tile = format_tile_fpath(tilename_dpath, tilename, tdem_dem_fpath)
    gdal_regrid(tdem_dem_fpath, tdem_dem_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['tdem_dem'] = tdem_dem_tile

    tdem_hem_tile = format_tile_fpath(tilename_dpath, tilename, tdem_hem_fpath)
    gdal_regrid(tdem_hem_fpath, tdem_hem_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['tdem_hem'] = tdem_hem_tile
    
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

    dem_fw = fdem_file.replace('F.tif', '_Fw.tif')
    dem_mw = mask_file.replace('M.tif', '_Mw.tif')

    if not os.path.isfile(dem_fw):
        fdem_file, mask_file = filter_tandemx_noise(tdem_dem_tile, 
                                                    tdem_hem_tile, 
                                                    tdem_com_tile, 
                                                    n_iter=1)  
    if not os.path.isfile(dem_fw):
        filter_water(fdem_file, mask_file, lcm_lwm_fn, esa_lwm_fn, wbm_lwm_fn)

    ds['tdem_dem_fw'] = dem_fw
    ds['tdem_dem_mw'] = dem_mw
    #os.remove(fdem_file)
    #os.remove(dem_mw)

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
   
    egm08_tile = format_tile_fpath(tilename_dpath, tilename, egm08_fpath)
    gdal_regrid(egm08_fpath, egm08_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['egm08'] = egm08_tile
   
    egm96_tile = format_tile_fpath(tilename_dpath, tilename, egm96_fpath)
    gdal_regrid(egm96_fpath, egm96_tile, xmin, ymin, xmax, ymax, xres, yres, mode='num')
    ds['egm96'] = egm96_tile

    # label_kmeans(dsm_path=edem_demw84_tile, dtm_path=pdem_tile, n_clusters=2, slope=False, overwrite=False)
    # label_kmeans(dsm_path=edem_demw84_tile, dtm_path=ldar_tile, n_clusters=2, slope=False, overwrite=False)

    # label_kmeans(dsm_path=edem_demw84_tile, dtm_path=pdem_tile, n_clusters=2, slope=True, overwrite=False)
    # label_kmeans(dsm_path=edem_demw84_tile, dtm_path=ldar_tile, n_clusters=2, slope=True, overwrite=False)

    # label_slope_adathresh(dsm_path=edem_demw84_tile, dtm_path=pdem_tile, overwrite=False)
    # label_slope_adathresh(dsm_path=edem_demw84_tile, dtm_path=ldar_tile, overwrite=False)

    # label_lcmultithresh_slope(dtm_path=pdem_tile, landcover_path=esawc_tile, percentile=75, overwrite=False)
    # label_lcmultithresh_slope(dtm_path=ldar_tile, landcover_path=esawc_tile, percentile=75, overwrite=False)

    # label_lcmultithresh(dtm_path=pdem_tile, landcover_path=esawc_tile, percentiles=[60, 75, 90], overwrite=False)
    # label_lcmultithresh(dtm_path=ldar_tile, landcover_path=esawc_tile, percentiles=[60, 75, 90], overwrite=False)

    # label_landdata(dsm_path=edem_demw84_tile, dtm_path=pdem_tile, landcover_path=esawc_tile, overwrite=False)
    # label_landdata(dsm_path=edem_demw84_tile, dtm_path=ldar_tile, landcover_path=esawc_tile, overwrite=False)
   
    # label_multiclass(dsm_path=edem_demw84_tile, dtm_path=pdem_tile, overwrite=False)
    # label_multiclass(dsm_path=edem_demw84_tile, dtm_path=ldar_tile, overwrite=False)

    # label_normthresh(dsm_path=edem_demw84_tile, dtm_path=pdem_tile, threshold=0.5, overwrite=False, norm=True)
    # label_normthresh(dsm_path=edem_demw84_tile, dtm_path=ldar_tile, threshold=0.5, overwrite=False, norm=False)

    # label_normthresh(dsm_path=edem_demw84_tile, dtm_path=pdem_tile, threshold=0.5, overwrite=False, norm=True)
    # label_normthresh(dsm_path=edem_demw84_tile, dtm_path=ldar_tile, threshold=0.5, overwrite=False, norm=False)

    # label_height_adathresh(dsm_path=edem_demw84_tile, dtm_path=pdem_tile, dynamic=True, base_threshold=0.5, overwrite=False)
    # label_height_adathresh(dsm_path=edem_demw84_tile, dtm_path=ldar_tile, dynamic=True, base_threshold=0.5, overwrite=False)

    # label_height_adathresh(dsm_path=edem_demw84_tile, dtm_path=pdem_tile, dynamic=False, base_threshold=0.5, overwrite=False)
    # label_height_adathresh(dsm_path=edem_demw84_tile, dtm_path=ldar_tile, dynamic=False, base_threshold=0.5, overwrite=False)

    # Define file paths for new terrain derivatives
    # tile_slp = edem_demw84_tile.replace('.tif', '_slp.tif')
    # tile_tpi = edem_demw84_tile.replace('.tif', '_tpi.tif')
    # tile_tri = edem_demw84_tile.replace('.tif', '_tri.tif')
    # tile_roughness = edem_demw84_tile.replace('.tif', '_rgx.tif')

    # # Compute Slope
    # if not os.path.isfile(tile_slp):
    #     dem_derivative(fi=edem_demw84_tile, fo=tile_slp, mode='slope')
    # ds['edem_slp'] = tile_slp

    # # Compute TPI
    # if not os.path.isfile(tile_tpi):
    #     dem_derivative(fi=edem_demw84_tile, fo=tile_tpi, mode='TPI')
    # ds['edem_tpi'] = tile_tpi

    # # Compute TRI
    # if not os.path.isfile(tile_tri):
    #     dem_derivative(fi=edem_demw84_tile, fo=tile_tri, mode='TRI')
    # ds['edem_tri'] = tile_tri

    # # Compute Roughness
    # if not os.path.isfile(tile_roughness):
    #     dem_derivative(fi=edem_demw84_tile, fo=tile_roughness, mode='roughness')
    # ds['edem_rgx'] = tile_roughness

    edem_grid_tile = f'{tilename_dpath}/{tilename}_EDEM_GRID.tif'
    print('*'*60)
    print(edem_grid_tile)
    print('-*-'*60)
    raster_calc(edem_dem_tile, edem_demw84_tile, "subtract", edem_grid_tile)
    ds['edem_grid'] = edem_grid_tile

    yaml_tile = os.path.join(tilename_dpath, f'{tilename}_ds.yaml')
    # csv_tile = os.path.join(tilename_dpath, f'{tilename}_ds.csv')
    # dsf = pd.DataFrame(ds)
    # dsf.to_csv(csv_tile, index=False)
    uops.write_yaml(ds, yaml_tile)
    print('yaml_tile:', yaml_tile)
    print('Already exists!!')
    
    print(ds)

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
    retile_datasets(
        ds_tiles_dpath, tilename, xmin, ymin, xmax, ymax, xres, yres,
        tdem_dem_fpath, tdem_hem_fpath, tdem_wam_fpath, tdem_com_fpath, 
        cdem_wbm_fpath, esawc_fpath, dtm_fpath,
        pdem_fpath, cdem_dem_fpath, edem_dem_fpath,egm08_fpath,edem_edem_W84_fpath,
        egm96_fpath,edem_lcm_fpath,s1_fpath, s2_fpath
    )

