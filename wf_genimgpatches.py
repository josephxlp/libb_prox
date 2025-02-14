
import os 
import pandas as pd 
import time 
from upaths import tilenames_full,TILES12_DPATH,tilenames_all,PATCHES_XDPATH
from utilspatches import (load_patch_params,load_variables,
                          filter_by_tilename,generate_tiles,get_tile_dict)
#from uvars import vending_all,nending_all 

def create_meta_df(tilenames,TILES12_DPATH,tilename,vending_all,nending_all):
    dlist = []
    for i in range(len(tilenames)):
        #if i > 0: break
        tilename = tilenames[i]
        filtered_dict,tile_dpath,names, paths = get_tile_dict(TILES12_DPATH,tilename,vending_all,nending_all)
        dlist.append({'tile':tilename,'names':names, 'paths':paths, 'tdpath':tile_dpath})

    dd = pd.DataFrame(dlist)
    return dd 

s1_ending = ['S1.tif']
s2_ending = ['S2.tif'] 
tar_ending =  ['NegroAOIDTM.tif', 'multi_DTM_LiDAR.tif','edem_W84.tif']

s1_names = ['s1']
s2_names = ['s2']
tar_names = ['pdem','ldem', 'edem']

nending_all =  tar_names+s1_names+s2_names
vending_all = tar_ending+s1_ending+s2_ending


X = 12 
ps = int(256 * 12) #1,4,8 12
tilenames = tilenames_all#tilenames_full
tilenames = ['N10E105','S01W063']


if __name__ == '__main__':
    outdir = f"{PATCHES_XDPATH}/{X}"
    os.makedirs(outdir, exist_ok=True)
    dd = create_meta_df(tilenames,TILES12_DPATH,tilenames,vending_all,nending_all)

    ti = time.perf_counter()

    for tname in tilenames:

        ftile,fnames,fpaths, fwdir = filter_by_tilename(dd, tname=tname)
        tile_odpath = os.path.join(outdir,str(ps),tname)
        os.makedirs(tile_odpath, exist_ok=True)
        print(ftile, tname)
        for i in range(len(fnames)):
            ta = time.perf_counter()
            fpath,fname = fpaths[i],fnames[i]
            var_dpath = os.path.join(tile_odpath, fname)
            os.makedirs(var_dpath,exist_ok=True)
            generate_tiles(fpath, var_dpath, ps,ps,ps,ps, save_tiles=True, overwrite=False)
            tb = time.perf_counter() - ta
            print(f'RUN.TIME {tb/60} mins @{fname} {var_dpath}')

    tf = time.perf_counter() - ti 
    print(f'RUN.TIME {tf/60} mins')

