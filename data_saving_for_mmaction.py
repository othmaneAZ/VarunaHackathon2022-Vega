import numpy as np
import os
import cv2
import glob
import rasterio
import geopandas
from rasterio.features import rasterize
from joblib import Parallel, delayed
import multiprocessing


def get_farm_crop(geometry, profile):
    """
    This function takes as input a polygon (geometry) and a raster profile and outputs a mask of the farm land and 
    the bbox coordinates (in the coordinate system of the raster image).
    """
    mask = rasterize(shapes=[(geometry, 1)], out_shape=(profile['width'], profile['height']), transform=profile['transform'])
    xs, ys = np.where(mask == 1)
    x_min, y_min, x_max, y_max = xs.min(), ys.min(), xs.max(), ys.max()
    bbox = [x_min, y_min, x_max, y_max]
    
    return mask[x_min:x_max, y_min:y_max], bbox


#LABELS = geopandas.read_file(TRAIN_LABEL_FILE)
#labels = [int(x)-1 for x in LABELS['crop_type']]
#geometry = [row for row in LABELS['geometry']]

def save_patch(polygon, i):
    OUTPUT = '/data/stars/user/mguermal/varuna/frames_varuna_test/'
    IMG_FOLDER = '/data/stars/user/mguermal/varuna/VarunaHackathon2022/sentinel-2-image/'
    all_imgs = glob.glob(IMG_FOLDER+'*/*/IMG_DATA/*_TCI.jp2')
    for index, img in enumerate(all_imgs):  #72 --> 2021 
        img = rasterio.open(img) 
        profile = img.profile
    #       image_patch, masks = get_farm_crop(geometry, profile)
        
        tci = img.read()
        mask, (x_min, y_min, x_max, y_max) = get_farm_crop(polygon, profile)
        tci_r = tci[0][x_min:x_max, y_min:y_max].astype(float)
        tci_g = tci[1][x_min:x_max, y_min:y_max].astype(float)
        tci_b = tci[2][x_min:x_max, y_min:y_max].astype(float)
        
        image_patch = np.stack([tci_r, tci_g, tci_b], axis=2)
        full_path = OUTPUT+'/'+str(i)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        try:
            image_patch = cv2.resize(image_patch, (224,224))
            cv2.imwrite(full_path+'/'+'img_%05d.jpg' %index, image_patch)
        except:
            print('error index %05d' %i)


if __name__ == '__main__':
    
    #paths to change for nef
    IMG_FOLDER = '/data/stars/user/mguermal/varuna/VarunaHackathon2022/sentinel-2-image/'
    TRAIN_LABEL_FILE = '/data/stars/user/mguermal/varuna/VarunaHackathon2022/testing_area/testdata.shp'
    OUTPUT = '/data/stars/user/mguermal/varuna/frames_varuna/'
    all_imgs = glob.glob(IMG_FOLDER+'*/*/IMG_DATA/*_TCI.jp2')
    
    LABELS = geopandas.read_file(TRAIN_LABEL_FILE)
    #labels = [int(x)-1 for x in LABELS['crop_type']]
    geometry = [row for row in LABELS['geometry']]
    indexes = np.arange(len(geometry))
    
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(save_patch)(polygon, index) for (polygon, index) in zip(geometry, indexes))
    
    
    
