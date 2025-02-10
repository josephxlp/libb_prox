from uvars import tilenames_lidar,RES_DPATH, tilenames#,nending_all,ftnames
# from uvars import aux_ending30,s1_ending30,s2_ending30,tar_ending30
# from uvars import aux_ending90,s1_ending90,s2_ending90,tar_ending90
# from uvars import aux_ending12,s1_ending12,s2_ending12,tar_ending12
from uvars import vending_all
from ud_tilepartquets import tile_files_to_parquet_parallel
import time 
#import argparse

# # 90:1, 30:1 12:1
dataname = "bylabel"
X=12
#X=30
#X=90

# if X == 12:
#     tar_ending,aux_ending,s1_ending,s2_ending = aux_ending12,s1_ending12,s2_ending12,tar_ending12

# elif X == 30:
#     tar_ending,aux_ending,s1_ending,s2_ending = aux_ending30,s1_ending30,s2_ending30,tar_ending30

# elif X == 90:
#     tar_ending,aux_ending,s1_ending,s2_ending = aux_ending90,s1_ending90,s2_ending90,tar_ending90

# tar_ending,aux_ending,s1_ending,s2_ending = aux_ending12,s1_ending12,s2_ending12,tar_ending12
# #tilenames = tilenames_lidar
# vending_all = tar_ending+aux_ending+s1_ending+s2_ending

if __name__ == '__main__':
    ti = time.perf_counter()

    tile_files_to_parquet_parallel(tilenames, RES_DPATH, X, vending_all,dataname)#, nending_all, ftnames)
    tf = time.perf_counter() - ti 
    print(f'{tf/60} min(s)')

# def parse_args():
#     parser = argparse.ArgumentParser(description='Run the tile files to parquet conversion process.')
#     parser.add_argument('--X', type=int, choices=[12, 30, 90], required=True, help='Set the X value (12, 30, or 90)')
#     return parser.parse_args()


# def main():
#     args = parse_args()
#     X = args.X

#     # Set the appropriate file endings based on X
#     if X == 12:
#         tar_ending, aux_ending, s1_ending, s2_ending = aux_ending12, s1_ending12, s2_ending12, tar_ending12
#     elif X == 30:
#         tar_ending, aux_ending, s1_ending, s2_ending = aux_ending30, s1_ending30, s2_ending30, tar_ending30
#     elif X == 90:
#         tar_ending, aux_ending, s1_ending, s2_ending = aux_ending90, s1_ending90, s2_ending90, tar_ending90

#     tilenames = tilenames_lidar
#     vending_all = tar_ending + aux_ending + s1_ending + s2_ending

#     # Run the parallel process to convert tile files to parquet
#     ti = time.perf_counter()
#     tile_files_to_parquet_parallel(tilenames, RES_DPATH, X, vending_all)
#     tf = time.perf_counter() - ti 
#     print(f'{tf/60} min(s)')

# # Run the main function when the script is executed
# if __name__ == '__main__':
#     main()


