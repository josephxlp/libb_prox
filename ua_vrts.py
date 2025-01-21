
from os.path import join,exists, basename,isfile
from os import system
import glob
import numpy as np 
from concurrent.futures import ProcessPoolExecutor
import yaml
import time
import subprocess


def filter_x_isin_list(mylist, x):
    return [i for i in mylist if x in i]


def filter_x_notin_list(mylist, x):
     return [i for i in mylist if x not in i]


def get_x_from_list(mylist, x):
    return [i for i in mylist if x in i][0]

def get_tilename_from_tdem_basename(fpath):
    return basename(fpath).split('_')[4]


def get_tilename_from_edem_basename(fpath):
    return basename(fpath).split('_')[3]


def get_tilename_from_edembasename_gpkg(gpkg_path):
    return basename(gpkg_path).split('_')[3]


def get_files_recursively(dir):
    # imporove on this upto level 5 if 
    fs = glob.glob(f'{dir}/*/*/*.tif', recursive=True)
    if len(fs) == 0:
        fs = glob.glob(f'{dir}/*/*.tif', recursive=True)
    return fs

def get_edem_var_files(dir):
    edem_fpaths = get_files_recursively(dir)
    edem_fpaths = filter_x_notin_list(edem_fpaths, 'PREVIEW')
    
    edem_egm = get_x_from_list(edem_fpaths, 'EGM')
    edem_w84 = get_x_from_list(edem_fpaths, 'W84')
    edem_edm = get_x_from_list(edem_fpaths, 'EDM')
    edem_hem = get_x_from_list(edem_fpaths, 'HEM')
    edem_lcm = get_x_from_list(edem_fpaths, 'LCM')
    edem_hsd = get_x_from_list(edem_fpaths, 'HSD')
    return edem_egm, edem_w84, edem_edm, edem_hem, edem_lcm, edem_hsd



def gen_vrt_params(dir_VRTs, varnames, dname, allfiles):
    var_dict = dict()
    for i in range(len(varnames)):
        txt_path = join(dir_VRTs, f'{dname}_{varnames[i]}.txt')
        vrt_path = join(dir_VRTs, f'{dname}_{varnames[i]}.vrt')
        count_lists = sum(isinstance(item, list) for item in allfiles)
        if count_lists > 1:
            files = allfiles[i]
            var_dict[varnames[i]] = {
                'txt': txt_path, 'vrt': vrt_path, 'files': files}
        else:
            if allfiles:   
                var_files = filter_x_isin_list(
                    allfiles, f'{varnames[i]}.tif')
                var_dict[varnames[i]] = {
                    'txt': txt_path, 'vrt': vrt_path, 'files': var_files}
            else:
                files = 'NoPathProvided'
                var_dict[varnames[i]] = {
                    'txt': txt_path, 'vrt': vrt_path, 'files': files}
                print('Warning:', 'all files is empty')
    print('gen_vrt_params')
    return var_dict

def get_txt_vrt_files_by_var(yaml_data, gdict, gdict_var):
    vrt = yaml_data[gdict][gdict_var]['vrt']
    txt = yaml_data[gdict][gdict_var]['txt']
    files = yaml_data[gdict][gdict_var]['files']
    return vrt, txt, files

# write test for this with proper yaml data
def get_all_VRT_TXT_FILE_paths(yaml_data):
    VRT_paths, TXT_paths, FILE_paths = [], [], []
    for group_dict in list(yaml_data.keys()):
        for gdict_var in list(yaml_data[group_dict].keys()):
            vrt, txt, files = get_txt_vrt_files_by_var(
                yaml_data, group_dict, gdict_var)
            VRT_paths.append(vrt)
            TXT_paths.append(txt)
            FILE_paths.append(files)
    print('get_all_VRT_TXT_FILE_paths')
    return VRT_paths, TXT_paths, FILE_paths

def buildVRT_from_list(txt_i, vrt_i, files):
    write_list_to_txt(txt_i, files) 
    buildVRT(txt_i, vrt_i)

def write_list_to_txt(txt, llist):
    with open(txt, 'w') as txt_write:
        for i in llist:
            txt_write.write(i+'\n')
    print('write_list_to_txt')

def buildVRT(txt, vrt):
    cmd = ['gdalbuildvrt', '-overwrite', '-input_file_list', txt, vrt]
    try:
        subprocess.run(cmd, check=True)
        print("VRT file created successfully at:", vrt)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
    except FileNotFoundError:
        print(f"Error: One or more files listed in '{txt}' do not exist.")
    except Exception as e:
        print("An unexpected error occurred:", e)


def process_lfiles(ldar_files, ldar_wrapped_dpath, epsg, xres):
    # wrap_lidar_files

    ldar_wrapped_files = [join(ldar_wrapped_dpath,basename(fi)) for fi in ldar_files]

    epsg_l = np.repeat(epsg, len(ldar_wrapped_files))
    res_l = np.repeat(xres, len(ldar_wrapped_files))

    with ProcessPoolExecutor() as ppe:
        ppe.map(warp_raster, ldar_files, ldar_wrapped_files, 
                res_l, epsg_l)
    return ldar_wrapped_files

def warp_raster(fi, fo, res, tepsg):
    if not exists(fo):
        cmd = f'gdalwarp -wo num_threads=all -co compress=deflate \
            -t_srs {tepsg} -tr {res} {res} {fi} {fo}'
        system(cmd)
    else:
        print('Already exisits')

def write_yaml(yaml_data, yaml_file_path):
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)
    print('write_yaml')


def read_yaml(filename):
    with open(filename, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        print('read_yaml')
        return yaml_data


def makedic(params_files, params_names):
    params_dict = {}
    for file, name in zip(params_files, params_names):
        if isfile(file):
            params_dict[name] = file
    print('makedic')
    return params_dict


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f'Elapsed time: {(end - start) * 1000:.3f}ms')
    return wrapper

def lenlist(ll):
    print(len(ll))

