from uvars import tilenames_mkd, tilenames_tls,tilenames_rgn
from upaths import WDIR_TILESX
from ud_tilepartquets import tile_files_to_parquet_parallel
import time 

#get my filled dems in the tiles folder 
s1_ending = ['S1X.tif']
s2_ending = ['S2.tif'] 
tar_ending =  ['NegroAOIDTM.tif', 'multi_DTM_LiDAR.tif','edem_W84.tif','cdem_DEM.tif','edem_EGM.tif']

s1_names = ['s1']
s2_names = ['s2']

s1_fnames = ['vv','vh']
s2_fnames = ['red','green','blue']
tar_names = ['pdem','ldem', 'edeme','cdem','edemo']

vending_all = tar_ending+s1_ending+s2_ending
nending_all =  tar_names+s1_names+s2_names
features_col = s1_fnames +s2_fnames
X=12

dataname = "core"
#if dataname == "roia":
tilenames = tilenames_mkd + tilenames_tls + tilenames_rgn
RES_DPATH = WDIR_TILESX
if __name__ == '__main__':
    ti = time.perf_counter()
    print(RES_DPATH)

    tile_files_to_parquet_parallel(tilenames, RES_DPATH, X, vending_all,dataname)#, nending_all, ftnames)
    tf = time.perf_counter() - ti 
    print(f'{tf/60} min(s)')