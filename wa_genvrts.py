from os import makedirs,cpu_count
from os.path import join,basename
import glob
from upaths import (WDIR,pband_k,preview_k,single_path_names,
                    egm08_fpath,egm96_fpath,
                    ds_dpath,cdem_dpath,edem_dpath,dtm_dpath,
                    dtm_wrapped_dpath,dsm_wrapped_dpath,
                    esa_path,pband_dpath,tdemx_dpath,
                    edem_vars,cdem_vars,tdem_vars,     
                    epsg,xres,multi_vars,
                    SENTINEL1_DPATH, SENTINEL2_DPATH)
import utilsvrt as uops 
from concurrent.futures import ProcessPoolExecutor
import time 
cpus = int(cpu_count() * 0.75)


if __name__ == '__main__':
    ti = time.perf_counter()
    TILES12_dpath = join(WDIR, 'TILES12')
    VRTs_dpath = join(WDIR, 'VRTs')
    CONFIG_dpath = join(WDIR, 'CONFIG')

    makedirs(VRTs_dpath, exist_ok=True)
    makedirs(CONFIG_dpath, exist_ok=True)
    makedirs(TILES12_dpath, exist_ok=True)

    step0_yaml_fpath = f'{CONFIG_dpath}/step0_ds_to_main_vars.yaml'
    step1_yaml_fpath = f'{CONFIG_dpath}/step1_main_vars_vrts.yaml'

    print('pband processing... ')
    pband_files = glob.glob(f'{pband_dpath}/*/*.tif')
    pband_files = uops.filter_x_isin_list(pband_files, pband_k)
    pband_fpath = pband_files[0]
    single_path_files = [egm08_fpath, egm96_fpath,pband_fpath]

    print('edem_files processing... ')
    edem_files = glob.glob(f'{edem_dpath}/*/*/*.tif', recursive=True);# print(len(edem_files))
    edem_files = uops.filter_x_notin_list(edem_files, preview_k); #print(len(edem_files))
    edem_dict = uops.gen_vrt_params(VRTs_dpath, edem_vars,'edem',edem_files)

    print('cdem_files processing... ')
    cdem_files = glob.glob(f'{cdem_dpath}/*/*/*.tif', recursive=True);# print(len(cdem_files))
    cdem_files = uops.filter_x_notin_list(cdem_files, preview_k);# lenlist(cdem_files)
    cdem_dict = uops.gen_vrt_params(VRTs_dpath, cdem_vars,'cdem',cdem_files)

    print('tdem_files processing... ')
    tdemx_files = glob.glob(f'{tdemx_dpath}//**/*.tif', recursive=True); #print(len(tdemx_files))
    tdemx_files = uops.filter_x_notin_list(tdemx_files, preview_k);# print(len(tdemx_files))
    tdem_dict = uops.gen_vrt_params(VRTs_dpath, tdem_vars,'tdem',tdemx_files)

    print('esa_files processing... ')
    esa_files = glob.glob(f'{esa_path}//**/*.tif', recursive=True)
    #fs = glob.glob(f'{ethchm_and_esawc_dpath}/*/*.tif'); #lenlist(fs)
    #ethchm_files = uops.filter_x_isin_list(fs, 'ETH.tif');#lenlist(ethchm_files)
    #esawc_files = uops.filter_x_isin_list(fs, 'ESAWC.tif')#;lenlist(esawc_files)
    #wsf2019_files = glob.glob(f'{wsf2019_dpath}/*.tif')#; lenlist(wsf2019_files)
    #gfc2020_files = glob.glob(f'{gfc2020_dpath}/*.tif')#; lenlist(gfc2020_files)
    print('dtm_files processing... ')
    dtm_files = glob.glob(f'{dtm_dpath}/*/*.tif', recursive=True)
    dtm_dils = [i for i in dtm_files if 'ESTONIA' not in i]
    print(dtm_dpath)
    print(f'{dtm_dpath}/*/*.tif')
    print('dtm_files::::::')
    uops.lenlist(dtm_files)
    makedirs(dtm_wrapped_dpath,exist_ok=True)
    ldar_wrapped_files = uops.process_lfiles(
        dtm_files, dtm_wrapped_dpath, epsg, xres)

    #dtm_files = os.listdir(dsm_wrapped_dpath)
    print('Reprojected Lidar:', len(ldar_wrapped_files))
    print(ldar_wrapped_files)

    print('S1 and S2 processing... ')
    s1_files =  glob.glob(f'{SENTINEL1_DPATH}/*/*.tif', recursive=True); print(len(s1_files))
    s2_files =  glob.glob(f'{SENTINEL2_DPATH}/*/*.tif', recursive=True); print(len(s2_files))


    mfiles = [esa_files, ldar_wrapped_files,s1_files,s2_files]
    # #mfiles = [ethchm_files, esawc_files , wsf2019_files, gfc2020_files,
    # #         dtm_wrapped_files]#,dsm_wrapped_files]

    mdict = uops.gen_vrt_params(VRTs_dpath, multi_vars,'multi',mfiles)
    #for i in multi_vars: lenlist(mdict[i]['files'])

    yaml_data = {'mdict': mdict,'cdem_dict': cdem_dict,
             'edem_dict': edem_dict,'tdem_dict': tdem_dict}
    uops.write_yaml(yaml_data, step0_yaml_fpath)
    yaml_data = uops.read_yaml(step0_yaml_fpath)

    VRT_paths, TXT_paths, FILE_paths = uops.get_all_VRT_TXT_FILE_paths(yaml_data)
    print(len(VRT_paths),len(TXT_paths), len(FILE_paths))

    print('building vrts processing... ')
    with ProcessPoolExecutor(cpus) as ppe:
        #for t,p in zip(TXT_paths, FILE_paths):
            #galBuildVRT_from_list(t,p)
        # ppe.map(galBuildVRT_from_list, TXT_paths,FILE_paths)
        #ppe.map(build_vrt_from_list, TXT_paths,VRT_paths, FILE_paths)
        ppe.map(uops.buildVRT_from_list, TXT_paths,VRT_paths, FILE_paths)

    VRT_names = [basename(VRT_paths[i][:-4]) for i in range(len(VRT_paths))]
    params_files = single_path_files + VRT_paths 
    params_names = single_path_names + VRT_names
    params_dict = uops.makedic(params_files,params_names)
    uops.write_yaml(params_dict, step1_yaml_fpath)
    tf = time.perf_counter() - ti
    print(f'RUN.TIME {tf/60} mins')








    


