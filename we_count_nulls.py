import os
import pandas as pd
from osgeo import gdal
import rasterio

from upaths import tilenames_full # tilenames_train  instead
from upaths import PATCHE256_DPATH,CONFIG_PATCHE256_DPATH,columns_to_check
from concurrent.futures import ProcessPoolExecutor


# Assuming these are defined elsewhere in your code
# from upaths import tilenames_full, PATCHE256_DPATH, CONFIG_PATCHE256_DPATH, columns_to_check

def create_dataframe_from_directory(directory):
  # Dictionary to hold lists of file paths for each subdirectory
  data = {}

  # Walk through the directory
  for root, dirs, files in os.walk(directory):
      # Get the relative path of the current directory
      relative_path = os.path.relpath(root, directory)

      # Skip the root directory itself
      if relative_path == ".":
          continue

      # Initialize a list for the current subdirectory if not already present
      if relative_path not in data:
          data[relative_path] = []

      # Add full paths of .tif files to the list
      for file in files:
          if file.endswith('.tif'):
              full_path = os.path.join(root, file)
              data[relative_path].append(full_path)

  # Create a DataFrame from the dictionary
  if data:
      df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
  else:
      df = pd.DataFrame()

  return df

def count_nulls_in_rasters(df, columns):
  # Add new columns for null counts
  for column in columns:
      if column in df.columns:
          null_counts = []
          for file_path in df[column].dropna():
              try:
                  with rasterio.open(file_path) as src:
                      # Read the data
                      data = src.read(1)
                      # Count nulls (assuming nulls are represented by NaN or a specific nodata value)
                      nodata_value = src.nodata
                      if nodata_value is not None:
                          null_count = (data == nodata_value).sum()
                      else:
                          null_count = pd.isnull(data).sum()
                      null_counts.append(null_count)
              except Exception as e:
                  print(f"Error processing file {file_path}: {e}")
                  null_counts.append(None)

          # Add the null counts to the DataFrame
          df[f'{column}_null_count'] = pd.Series(null_counts)

  return df

def count_nulls_by_variablelist(directory_path, columns_to_check, csvpath):
    if not os.path.isfile(csvpath):
        df = create_dataframe_from_directory(directory_path)
        if df.empty:
            print(f"No .tif files found in {directory_path}")
        else:
            #print(df.head())
            df_with_null_counts = count_nulls_in_rasters(df, columns_to_check)
            df_with_null_counts.to_csv(csvpath, index=False)
            print(f'Processed {directory_path} and saved to {csvpath}')
    else:
        print('already exist')


from utilstimer import Timer 
from upaths import CONFIG_PATH_DPATH
from upaths import tilenames_all

# Example usage:

ps = int(256 * 4)

timer = Timer()
CONFIG_PATCHE_X_DPATH = os.path.join(CONFIG_PATH_DPATH, f'PATCHES{ps}')
os.makedirs(CONFIG_PATCHE_X_DPATH,exist_ok=True)

tilenames = tilenames_all

if __name__ == '__main__':
    timer.time_start()
    with ProcessPoolExecutor() as PEX:
        for tilename in tilenames:
            directory_path = os.path.join(PATCHE256_DPATH, tilename)
            print(f"Processing directory: {directory_path}")
            csvpath = os.path.join(CONFIG_PATCHE_X_DPATH, f'{tilename}_patches_count_null.csv')

            PEX.submit(count_nulls_by_variablelist, directory_path, columns_to_check, csvpath)
    timer.time_end()