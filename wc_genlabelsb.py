import os 
import time 
import multiprocessing
from pprint import pprint
from osgeo import gdal, gdalconst


import ua_vrts as uops # 
#import utilsregrid as rops 
from upaths import step0_yaml_fpath, step1_yaml_fpath,WDIR_TILES12
from ub_tiles import process_tile

# add:
# scaler
# vfill (quick)
# grid from edem wgs and edem  in wzdev.ipynb 

gdal.UseExceptions()
mem_drv = gdal.GetDriverByName('MEM')
gtif_drv = gdal.GetDriverByName('GTiff')
vrt_drv = gdal.GetDriverByName("VRT")

names = ['tdem_dict', 'edem_dict', 'cdem_dict']
dicname = names[0] 

TILES12_DPATH = WDIR_TILES12
print('Runnign wb_gentiles.py')
print(f'Input files: \n{TILES12_DPATH}\n{step0_yaml_fpath}\n{step1_yaml_fpath}')
print('')
#print(dicname)


if __name__ == '__main__':
   
    ti = time.perf_counter()
    bpaths = uops.read_yaml(step0_yaml_fpath) 
    gpaths = uops.read_yaml(step1_yaml_fpath)
    os.makedirs(TILES12_DPATH,exist_ok=True)
    os.chdir(TILES12_DPATH)

    if dicname == 'tdem_dict':
        basefiles = bpaths['tdem_dict']['DEM']['files'] 
    elif dicname == 'cdem_dict':
        basefiles = bpaths['cdem_dict']['DEM']['files'] 
    elif dicname == 'edem_dict':
        basefiles = bpaths['edem_dict']['EGM']['files'] 

    #print(dicname)
    print(f'basefiles {len(basefiles)}')
    pprint(basefiles)
    print(f'basefiles {len(basefiles)}')

    tdem_dem_fpath = gpaths['tdem_DEM']
    tdem_hem_fpath = gpaths['tdem_HEM']
    tdem_wam_fpath = gpaths['tdem_WAM']  
    tdem_com_fpath = gpaths['tdem_COM']
    cdem_wbm_fpath = gpaths['cdem_WBM']

    dtm_fpath = gpaths['multi_DTM_LiDAR']
    esawc_fpath = gpaths['multi_ESAWC']
    pdem_fpath = gpaths['pband']
    cdem_dem_fpath = gpaths['cdem_DEM']
    edem_dem_fpath = gpaths['edem_EGM']
    edem_edem_W84_fpath = gpaths['edem_W84']
    edem_lcm_fpath = gpaths['edem_LCM']
    #wsfba_fpath = gpaths['wsfba']
    #wsfbf_fpath = gpaths['wsfbf']
    #wsfbh_fpath = gpaths['wsfbh']
    #wsfbv_fpath = gpaths['wsfbv']
    ####egm08_fpath = gpaths['egm08'] #@ put this back in
    #dsm_fpath = gpaths['multi_DSM_LiDAR']
    egm08_fpath = gpaths['egm08']
    egm96_fpath = gpaths['egm96']
    s1_fpath = gpaths['multi_S1']
    s2_fpath = gpaths['multi_S2']
   

    num_processes = int(multiprocessing.cpu_count() * 0.75)
    pool = multiprocessing.Pool(processes=num_processes)

    for i, basefile in enumerate(basefiles):
        print(f'{i}/{len(basefiles)} @{basefile}')
        pool.apply_async(
            process_tile, (basefile, TILES12_DPATH, tdem_dem_fpath, tdem_hem_fpath, 
                            tdem_wam_fpath, tdem_com_fpath, cdem_wbm_fpath, esawc_fpath, 
                            dtm_fpath,  pdem_fpath, cdem_dem_fpath, 
                            edem_dem_fpath,egm08_fpath,edem_edem_W84_fpath,egm96_fpath,
                            edem_lcm_fpath,s1_fpath, s2_fpath))
        
    pool.close()
    pool.join()
    # for i, basefile in enumerate(basefiles):
    #     if i > 0 : break
    #     print(f'{i}/{len(basefiles)} @{basefile}')
    #     process_tile(
    #         basefile, TILES12_DPATH, tdem_dem_fpath, tdem_hem_fpath, 
    #                         tdem_wam_fpath, tdem_com_fpath, cdem_wbm_fpath, esawc_fpath, 
    #                         dtm_fpath,  pdem_fpath, cdem_dem_fpath, 
    #                         edem_dem_fpath,egm08_fpath,edem_edem_W84_fpath,
    #                         egm96_fpath,edem_lcm_fpath,s1_fpath, s2_fpath)


    print("All tasks completed")
    tf = time.perf_counter() - ti
    print(f'run.time: {tf/60} min(s)')