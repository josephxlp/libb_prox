
import os 
from glob import glob 
from rublabels import ru_labels
from concurrent.futures import ProcessPoolExecutor


dtm_pathx = "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/TILES12/*/*_NegroAOIDTM.tif"
dsm_pathx = "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/TILES12/*/*_edem_W84.tif"
landcover_pathx = "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/TILES12/*/*_multi_ESAWC.tif"

if __name__ == '__main__':

    dtm_files = glob(dtm_pathx); print(len(dtm_files))
    dsm_files = glob(dsm_pathx);print(len(dsm_files))
    landcover_files = glob(landcover_pathx);print(len(landcover_files))

    with ProcessPoolExecutor(17) as PEX:

        for i, (dsm_path, dtm_path, landcover_path) in enumerate(zip(dsm_files, dtm_files, landcover_files)):
            #if i > 0: break
            print(dsm_path)

            PEX.submit(
                ru_labels,dsm_path,dtm_path,landcover_path
            )

