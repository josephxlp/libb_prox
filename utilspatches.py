import os 
import yaml
import time 
import pickle
import os
import itertools
import numpy as np
import rasterio as rio
from rasterio import windows
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, List, Tuple
import pandas as pd 


def get_raster_meta(path: str) -> Tuple[dict, int, int]:
  """Open a raster file and return its metadata, height, and width."""
  with rio.open(path) as ds:
      meta = ds.meta
      height = meta["height"]
      width = meta["width"]
      meta["crs"] = ds.crs
  return meta, height, width

def calculate_offsets(width: int, height: int, stride_x: int, stride_y: int) -> List[Tuple[int, int]]:
  """Calculate offsets for tiling based on the raster dimensions and stride."""
  X = range(0, width, stride_x)
  Y = range(0, height, stride_y)
  return list(itertools.product(X, Y))

def process_tile(path: str, col_off: int, row_off: int, tile_x: int, tile_y: int, output_folder: str, save_tiles: bool, overwrite: bool):
  """Process a single tile and save it if required."""
  tile_name = f"tile_{col_off}_{row_off}.tif"
  tile_path = os.path.join(output_folder, tile_name)

  if not overwrite and os.path.exists(tile_path):
      print(f"Tile {tile_name} already exists. Skipping...")
      return

  with rio.open(path) as ds:
      window = windows.Window(col_off=col_off, row_off=row_off, width=tile_x, height=tile_y)
      transform = windows.transform(window, ds.transform)
      meta = ds.meta.copy()
      meta.update({"width": window.width, "height": window.height, "transform": transform})

      tile_data = ds.read(window=window, boundless=True)
      if save_tiles:
          with rio.open(tile_path, "w", **meta) as outds:
              outds.write(tile_data)

def generate_tiles(path: str, output_folder: str, tile_x: int = 256, tile_y: int = 256, stride_x: int = 256, stride_y: int = 256, save_tiles: bool = True, overwrite: bool = False):
  """Generate tiles from a raster file and save them to the specified folder."""
  meta, height, width = get_raster_meta(path)
  offsets = calculate_offsets(width, height, stride_x, stride_y)

  cpus = max(1, int(os.cpu_count() * 0.8))  # Use 80% of available CPUs
  if not os.path.exists(output_folder) and save_tiles:
      os.makedirs(output_folder)

  with ProcessPoolExecutor(cpus) as executor:
      futures = [
          executor.submit(process_tile, path, col_off, row_off, tile_x, tile_y, output_folder, save_tiles, overwrite)
          for col_off, row_off in offsets
      ]
      for future in futures:
          future.result()


def filter_by_tilename(dd, tname='S01W063'):
    subdd =  dd[dd['tile'] == tname].values.tolist()[0]
    ftile,fnames,fpaths, fwdir = subdd[0],subdd[1], subdd[2], subdd[3]
    return ftile,fnames,fpaths, fwdir


def write_yaml(yaml_data, yaml_file_path):
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)
    print('write_yaml')


def read_yaml(filename):
    with open(filename, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        print('read_yaml')
        return yaml_data

def filter_dictionary_by_keys(dictionary, key_list):
    filtered_dict = {key: value for key, value in dictionary.items() if key in key_list}
    return filtered_dict

def split_dictionary_pair(data_dict):
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    return keys, values

def load_variables(i_wdir, tile, ps):
    yaml_tile = f"{i_wdir}/{tile}/{tile}_ds.yaml"
    print(yaml_tile)
    wdir = f"{i_wdir}_patches{ps}"
    tilename = yaml_tile.split('/')[-2]
    return yaml_tile, wdir, tilename

def load_patch_params(yaml_tile, wdir, tilename, vrtvars):
    os.makedirs(wdir,exist_ok=True)
    data_dict = read_yaml(yaml_tile)
    # print(list(data_dict.keys()) == vrtvars)
    wdir_tile_dpath = os.path.join(wdir, tilename)
    os.makedirs(wdir_tile_dpath,exist_ok=True)
    data_dict_filter = filter_dictionary_by_keys(data_dict, vrtvars)
    names, paths = split_dictionary_pair(data_dict_filter)
    assert len(names) == len(paths), 'len(names) == len(paths) failed!!!'
    return names, paths, wdir_tile_dpath

def filter_patch_params(names, paths, wdir_tile_dpath, i):
    iname = names[i]
    ipath = paths[i]
    wdir_tile_dpath_var = os.path.join(wdir_tile_dpath, iname)
    os.makedirs(wdir_tile_dpath_var, exist_ok=True)
    return iname, ipath, wdir_tile_dpath_var
