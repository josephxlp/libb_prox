import os
import subprocess
import numpy as np
import rasterio
from sklearn.preprocessing import MinMaxScaler
from osgeo import gdal


import rasterio
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def scale_raster(input_raster, method="minmax"):
    output_raster = input_raster.replace('.tif', 'X.tif')
    # Check if the output file already exists
    if os.path.exists(output_raster):
        print(f"File already exists: {output_raster}")
        return
    
    # Open raster
    with rasterio.open(input_raster) as src:
        data = src.read().astype(np.float32)  # Read as float32 for scaling
        profile = src.profile  # Get metadata

    # Reshape for scaling
    n_bands, height, width = data.shape
    reshaped_data = data.reshape(n_bands, -1).T  # (pixels, bands)

    # Choose scaler
    if method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaling method. Use 'minmax' or 'standard'.")

    # Fit and transform
    scaled_data = scaler.fit_transform(reshaped_data)
    scaled_data = scaled_data.T.reshape(n_bands, height, width)  # Reshape back

    # Update metadata
    profile.update(dtype=rasterio.float32, nodata=None)

    # Save scaled raster
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(scaled_data.astype(np.float32))

    print(f"Scaled raster saved to {output_raster}")
    return output_raster




def dem_derivative(fi, fo, mode='slope'):
    valid_modes = ['hillshade', 'slope', 'aspect', 'TRI', 'TPI', 'roughness']
    
    if mode not in valid_modes:
        print(f"Invalid mode. Please choose one of: {', '.join(valid_modes)}")
        return
    
    subprocess.run(['gdaldem', mode, fi, fo])

def raster_calc(dsm_path, dtm_path, operation, output_path):

    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return
     
    # Abrir os arquivos raster
    with rasterio.open(dsm_path) as dsm, rasterio.open(dtm_path) as dtm:
        if dsm.shape != dtm.shape:
            raise ValueError("Os rasters DSM e DTM devem ter o mesmo tamanho.")

        dsm_data = dsm.read(1)  # Lê a primeira banda
        dtm_data = dtm.read(1)  # Lê a primeira banda

        # Realizar a operação desejada
        if operation == 'add':
            result_data = dsm_data + dtm_data
        elif operation == 'subtract':
            result_data = dsm_data - dtm_data
        else:
            raise ValueError("Operação inválida. Escolha 'add' ou 'subtract'.")

        # Configuração dos metadados para o arquivo de saída
        output_meta = dsm.meta.copy()
        output_meta.update({"dtype": "float32"})  # Garante compatibilidade

        # # Criar diretório de saída se não existir
        # os.makedirs(out_dir, exist_ok=True)
        # output_path = os.path.join(out_dir, f"{output_name}.tif")

        # Salvar o raster resultante
        with rasterio.open(output_path, "w", **output_meta) as dest:
            dest.write(result_data.astype(np.float32), 1)

        print(f"Arquivo salvo em: {output_path}")
