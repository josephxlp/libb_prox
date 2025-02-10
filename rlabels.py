import os 
import numpy as np
import rasterio
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


def compute_slope(dtm_path):
    with rasterio.open(dtm_path) as dtm_ds:
        dtm_data = dtm_ds.read(1, masked=True)
        x, y = np.gradient(dtm_data.filled(0), dtm_ds.res[0], dtm_ds.res[1])
        slope = np.sqrt(x**2 + y**2)
        return np.ma.masked_invalid(slope)

def label_kmeans(dsm_path, dtm_path, n_clusters=2, slope=False, overwrite=False):
    output_path = dtm_path.replace('.tif', '_label_kmeans.tif')
    if slope:
        output_path = dtm_path.replace('.tif', '_label_kmeans_slope.tif')

    if os.path.exists(output_path) and not overwrite:
        print(f"File {output_path} already exists. Use overwrite=True to replace it.")
        return

    with rasterio.open(dsm_path) as dsm_ds, rasterio.open(dtm_path) as dtm_ds:
        dsm_data = dsm_ds.read(1, masked=True)
        dtm_data = dtm_ds.read(1, masked=True)
        height_diff = np.ma.masked_invalid(dsm_data - dtm_data)

        features = [height_diff, dtm_data, dsm_data]
        if slope:
            dtm_slope = compute_slope(dtm_path)
            features.append(dtm_slope)

        stacked_features = np.ma.dstack(features)
        valid_pixels = ~height_diff.mask
        feature_values = stacked_features[valid_pixels].reshape(-1, len(features))

        scaler = StandardScaler()
        feature_values_scaled = scaler.fit_transform(feature_values)

        if feature_values_scaled.size == 0:
            raise ValueError("No valid pixels available for clustering.")

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(feature_values_scaled)

        labels = np.zeros_like(height_diff, dtype=np.uint8)
        labels[valid_pixels] = kmeans.labels_ + 1

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=labels.shape[0],
        width=labels.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=dsm_ds.crs,
        transform=dsm_ds.transform
    ) as dst_ds:
        dst_ds.write(labels, 1)

    print(f"K-Means classification mask saved to {output_path}")

# very expenise
def label_adathreshold(dsm_path, dtm_path, overwrite=False, window_size=255):

    mask_path = dtm_path.replace('.tif', '_label_adathreshold.tif')
    if os.path.exists(mask_path) and not overwrite:
        print(f"File {mask_path} already exists. Use overwrite=True to replace it.")
        return
    
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number for proper window centering.")

    with rasterio.open(dsm_path) as dsm_ds, rasterio.open(dtm_path) as dtm_ds:
        dsm_data = dsm_ds.read(1, masked=True)
        dtm_data = dtm_ds.read(1, masked=True)
        transform = dsm_ds.transform

        height_diff = np.ma.masked_invalid(dsm_data - dtm_data)

        dx, dy = np.gradient(dtm_data.filled(0))
        slope = np.sqrt(dx**2 + dy**2)
        slope = np.ma.masked_invalid(slope)  

        mask = np.zeros(dsm_data.shape, dtype=np.uint8)
        half_window = window_size // 2

        for i in range(half_window, dsm_data.shape[0] - half_window):
            for j in range(half_window, dsm_data.shape[1] - half_window):
                win_diff = height_diff[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]
                win_slope = slope[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]

                valid_win_diff = win_diff.compressed()
                valid_win_slope = win_slope.compressed()

                if valid_win_diff.size > 0:
                    local_height_threshold_high = np.percentile(valid_win_diff, 75)
                else:
                    local_height_threshold_high = height_diff[i, j]

                if valid_win_slope.size > 0:
                    local_slope_threshold = np.percentile(valid_win_slope, 50)
                else:
                    local_slope_threshold = slope[i, j]

                central_pixel_diff = height_diff[i, j]
                central_pixel_slope = slope[i, j]

                if central_pixel_diff <= local_height_threshold_high and central_pixel_slope <= local_slope_threshold:
                    mask[i, j] = 2  
                else:
                    mask[i, j] = 1  

    with rasterio.open(
        mask_path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=dsm_ds.crs,
        transform=transform
    ) as mask_ds:
        mask_ds.write(mask, 1)

    print(f"Local adaptive mask created and saved to {mask_path}")


def label_slope_adathresh(dsm_path, dtm_path, overwrite=False):
    mask_path = dtm_path.replace('.tif', '_label_slope_adathresh.tif')
    
    if os.path.exists(mask_path) and not overwrite:
        print(f"File {mask_path} already exists. Use overwrite=True to replace it.")
        return

    with rasterio.open(dsm_path) as dsm_ds, rasterio.open(dtm_path) as dtm_ds:
        dsm_data = dsm_ds.read(1, masked=True)
        dtm_data = dtm_ds.read(1, masked=True)
        transform = dsm_ds.transform

        height_diff = np.ma.masked_invalid(dsm_data - dtm_data)

        dx, dy = np.gradient(dtm_data.filled(0), dtm_ds.res[0], dtm_ds.res[1])
        slope = np.sqrt(dx**2 + dy**2)
        slope = np.ma.masked_invalid(slope)

        low_height_threshold = np.percentile(height_diff.compressed(), 25)
        high_height_threshold = np.percentile(height_diff.compressed(), 75)
        slope_threshold = np.percentile(slope.compressed(), 50)

        mask = np.full(dsm_data.shape, 2, dtype=np.uint8)  

        mask[(height_diff > high_height_threshold) | (slope > slope_threshold)] = 1  

    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=dsm_ds.crs,
        transform=transform
    ) as mask_ds:
        mask_ds.write(mask, 1)

    print(f"Simplified land cover mask created and saved to {mask_path}")


#######################
import numpy as np
import rasterio
import os


def compute_slope(dtm_path):
    with rasterio.open(dtm_path) as dtm_ds:
        dtm_data = dtm_ds.read(1, masked=True)
        x, y = np.gradient(dtm_data.filled(0), dtm_ds.res[0], dtm_ds.res[1])
        slope = np.sqrt(x**2 + y**2)
        return np.ma.masked_invalid(slope)


def label_lcmultithresh(dtm_path, landcover_path, percentiles=[60, 75, 90], overwrite=False):
    mask_base_path = landcover_path.replace('.tif', '_label_lcmultithresh')

    with rasterio.open(dtm_path) as dtm_ds, rasterio.open(landcover_path) as landcover_ds:
        dtm_data = dtm_ds.read(1, masked=True)
        landcover_data = landcover_ds.read(1, masked=True)
        transform = dtm_ds.transform
        res_x, res_y = dtm_ds.res

        # Compute slope
        slope = compute_slope(dtm_path)

        # Calculate adaptive thresholds for the given percentiles
        adaptive_thresholds = {p: np.percentile(slope.compressed(), p) for p in percentiles}

        print("Adaptive slope thresholds:")
        for p, threshold in adaptive_thresholds.items():
            print(f"  - {p}th percentile: {threshold:.2f}°")

        # Land cover classes
        low_classes = [30, 40, 60, 70, 80, 90, 95, 100]
        high_classes = [10, 20, 50]

        # Generate masks for different percentiles
        for p, slope_threshold in adaptive_thresholds.items():
            mask_path = f"{mask_base_path}_{p}percentile.tif"
            if os.path.exists(mask_path) and not overwrite:
                print(f"File {mask_path} already exists. Use overwrite=True to replace it.")
                continue

            mask = np.full(landcover_data.shape, 2, dtype=np.uint8)

            # Assign low ground (2) for low classes and low slope
            mask[np.isin(landcover_data, low_classes) & (slope <= slope_threshold)] = 2
            # Assign high ground (1) for high classes or high slope
            mask[np.isin(landcover_data, high_classes) | (slope > slope_threshold)] = 1

            # Write the mask to an output file
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=mask.shape[0],
                width=mask.shape[1],
                count=1,
                dtype=rasterio.uint8,
                crs=dtm_ds.crs,
                transform=transform
            ) as mask_ds:
                mask_ds.write(mask, 1)

            print(f"Slope-based land cover mask saved to {mask_path}")

    print("Process completed.")


def label_lcmultithresh_slope(dtm_path, landcover_path, percentile=75, overwrite=False):
    mask_base_path = landcover_path.replace('.tif', '_label_lcmultithresh_slope.tif')
    if os.path.exists(mask_base_path) and not overwrite:
        print(f"File {mask_base_path} already exists. Use overwrite=True to replace it.")
        return

    with rasterio.open(dtm_path) as dtm_ds, rasterio.open(landcover_path) as landcover_ds:
        dtm_data = dtm_ds.read(1, masked=True)
        landcover_data = landcover_ds.read(1, masked=True)
        transform = dtm_ds.transform
        res_x, res_y = dtm_ds.res

        # Compute slope
        slope = compute_slope(dtm_path)

        # Calculate adaptive threshold based on the percentile
        adaptive_threshold = np.percentile(slope.compressed(), percentile)

        print(f"Adaptive slope threshold (based on {percentile}th percentile): {adaptive_threshold:.2f}°")

        # Create binary mask (initialize with 2 for low ground)
        mask = np.full(landcover_data.shape, 2, dtype=np.uint8)

        # Lower elevation classes (flat areas)
        low_classes = [30, 40, 60, 70, 80, 90, 95, 100]
        mask[np.isin(landcover_data, low_classes) & (slope <= adaptive_threshold)] = 2

        # Higher elevation classes (forests, built-up areas, steep slopes)
        high_classes = [10, 20, 50]
        mask[np.isin(landcover_data, high_classes) | (slope > adaptive_threshold)] = 1

    # Write the mask to the output file
    with rasterio.open(
        mask_base_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=dtm_ds.crs,
        transform=transform
    ) as mask_ds:
        mask_ds.write(mask, 1)

    print(f"Adaptive slope-based land cover mask created and saved to {mask_base_path}")

import numpy as np
import rasterio
import os


def label_landdata(dsm_path, dtm_path, landcover_path, overwrite=False):
    mask_path = landcover_path.replace('.tif', '_label_landdata.tif')

    if os.path.exists(mask_path) and not overwrite:
        print(f"File {mask_path} already exists. Use overwrite=True to replace it.")
        return

    # Open the DSM, DTM, and land cover files using Rasterio
    with rasterio.open(dsm_path) as dsm_ds, rasterio.open(dtm_path) as dtm_ds, rasterio.open(landcover_path) as landcover_ds:
        dsm_data = dsm_ds.read(1, masked=True)
        dtm_data = dtm_ds.read(1, masked=True)
        landcover_data = landcover_ds.read(1, masked=True)
        transform = dsm_ds.transform
        crs = dsm_ds.crs

        # Filter out invalid values in DSM and DTM (mask values outside the valid range)
        dsm_data = np.ma.masked_outside(dsm_data, -999, 1000)  # Mask values outside the valid range
        dtm_data = np.ma.masked_outside(dtm_data, -999, 1000)

        # Calculate height difference
        height_diff = dsm_data - dtm_data

        # Create binary mask (initialize with 2 for low ground)
        mask = np.full(dsm_data.shape, 2, dtype=np.uint8)

        # Lower elevation classes (set to 0)
        low_classes = [30, 40, 60, 70, 80, 90, 95, 100]
        mask[np.isin(landcover_data, low_classes)] = 0

        # Higher elevation classes (set to 1)
        high_classes = [10, 20, 50]
        mask[np.isin(landcover_data, high_classes)] = 1

    # Write the mask to the output file using Rasterio
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform
    ) as mask_ds:
        mask_ds.write(mask, 1)

    print(f"Binary land cover mask created and saved to {mask_path}")


def label_multiclass(dsm_path, dtm_path, overwrite=False):
    mask_path = dtm_path.replace('.tif', '_label_multiclass.tif')
    
    if os.path.exists(mask_path) and not overwrite:
        print(f"File {mask_path} already exists. Use overwrite=True to replace it.")
        return

    # Open DSM and DTM files using Rasterio
    with rasterio.open(dsm_path) as dsm_ds, rasterio.open(dtm_path) as dtm_ds:
        dsm_data = dsm_ds.read(1, masked=True).astype(float)
        dtm_data = dtm_ds.read(1, masked=True).astype(float)

        # Calculate height difference and slope
        height_diff = dsm_data - dtm_data
        gradient_x, gradient_y = np.gradient(dtm_data)
        slope = np.sqrt(gradient_x**2 + gradient_y**2)

        # Compute dynamic thresholds based on DTM statistics
        dtm_mean = np.nanmean(dtm_data)
        dtm_std = np.nanstd(dtm_data)

        low_thresh = dtm_mean - 0.5 * dtm_std
        moderate_thresh = dtm_mean
        high_thresh = dtm_mean + 0.5 * dtm_std

        # Use np.ma.compressed() to remove the masked values before calculating the percentile
        slope_thresh = np.nanpercentile(np.ma.compressed(slope), 90)  # Adaptive slope threshold

        # Create binary mask (initialize with 2 for low ground)
        mask = np.full(dsm_data.shape, 2, dtype=np.uint8)

        # Set values based on height difference and slope
        mask[(height_diff < low_thresh) & (slope < slope_thresh * 0.5)] = 1  # Flat low areas
        mask[(height_diff > high_thresh) & (slope > slope_thresh)] = 1  # Steep areas

        # Write the mask to the output file using Rasterio
        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=rasterio.uint8,
            crs=dsm_ds.crs,
            transform=dsm_ds.transform
        ) as mask_ds:
            mask_ds.write(mask, 1)

    print(f"Binary land cover mask created and saved to {mask_path}")


def label_normthresh(dsm_path, dtm_path, threshold=0.5, overwrite=False, norm=True):
    # Modify the mask path based on whether normalization is applied
    norm_str = "_normy" if norm else "_normn"
    mask_path = dtm_path.replace('.tif', f'_label{norm_str}_thresh{threshold}.tif')

    if os.path.exists(mask_path) and not overwrite:
        print(f"File {mask_path} already exists. Use overwrite=True to replace it.")
        return

    with rasterio.open(dsm_path) as dsm_ds, rasterio.open(dtm_path) as dtm_ds:
        if dsm_ds is None or dtm_ds is None:
            raise FileNotFoundError("DSM or DTM file not found.")

        # Read the data
        dsm_data = dsm_ds.read(1)
        dtm_data = dtm_ds.read(1)

        # Calculate height difference
        height_diff = dsm_data - dtm_data

        # Apply normalization if `norm=True`
        if norm:
            normalised_diff = (height_diff - np.nanmean(height_diff)) / np.nanstd(height_diff)
        else:
            normalised_diff = height_diff

        # Create the mask with 0 for low values (ground) and 1 for high values (non-ground)
        mask = np.full(dsm_data.shape, 0, dtype=np.uint8)
        mask[np.isnan(dsm_data) | np.isnan(dtm_data)] = 2
        mask[(~np.isnan(dsm_data)) & (~np.isnan(dtm_data)) & (normalised_diff < threshold)] = 1

        # Write the mask to output file
        with rasterio.open(mask_path, 'w', driver='GTiff', count=1, dtype=rasterio.uint8,
                           width=dsm_ds.width, height=dsm_ds.height, crs=dsm_ds.crs, transform=dsm_ds.transform) as mask_ds:
            mask_ds.write(mask, 1)

    print(f"Mask created and saved to {mask_path}")



