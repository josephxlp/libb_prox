
from upaths import tilenames_full,TILES12_DPATH,tilenames_all
from utilspatches import (load_patch_params,load_variables,
                          filter_by_tilename,generate_tiles)
import os 
import pandas as pd 
import time 

names_at_patching_V1 = ['edem_demw84','ldtm','pdem','tdem_dem',
                        'egm08x','egm96x','s1x','s2x','tdem_hem','esawc','lwm',
                       'tdem_dem_fw','tdem_dem_mw']
vrtvars = names_at_patching_V1
i_wdir = TILES12_DPATH
ps = int(256 * 4) #1,4

tilenames = tilenames_all#tilenames_full
print(tilenames)
dlist = []
for i in range(len(tilenames)):
    #if i > 0: break
    tile = tilenames[i]
    yaml_tile, wdir, tilename = load_variables(i_wdir, tile, ps)
    names, paths, wdir_tile_dpath = load_patch_params(yaml_tile, wdir, tilename, vrtvars)
    dlist.append({'tile':tile,'names':names, 'paths':paths, 'tdpath':wdir_tile_dpath})

dd = pd.DataFrame(dlist)

if __name__ == '__main__':
    ti = time.perf_counter()

    for tname in tilenames:

        ftile,fnames,fpaths, fwdir = filter_by_tilename(dd, tname=tname)
        print(ftile, tname)
        for i in range(len(names)):
            ta = time.perf_counter()
            fpath,fname = fpaths[i],fnames[i]
            wdir_vtile_dpath = os.path.join(fwdir, fname)
            os.makedirs(wdir_vtile_dpath,exist_ok=True)
            generate_tiles(fpath, wdir_vtile_dpath, ps,ps,ps,ps, 
                           save_tiles=True, overwrite=False)
            tf = time.perf_counter() - ti 
            print(f'RUN.TIME {tf/60} mins @{fname} {fwdir}')

    tf = time.perf_counter() - ti 
    print(f'RUN.TIME {tf/60} mins')


# add count of nulls by tdem, pdem,ldem 