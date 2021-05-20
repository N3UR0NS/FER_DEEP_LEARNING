import csv
import os
from PIL import Image
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def csv2img(dataset_path):
    # We broke dataset file(.csv) into part so it can be uploaded to github
    csvs = [f for f in  os.listdir(dataset_path) if f.endswith('.csv')]

    csv_list = []
    for csv_file in sorted(csvs):
        print(f'--> {csv_file}')
        csv_list.append(pd.read_csv(os.path.join(dataset_path, csv_file)))
    
    df = pd.concat(csv_list, axis=0, ignore_index=True)

    emotions = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}

    for name in ('data/train', 'data/test-public', 'data/test-private'):
        for emotion in emotions:
            os.makedirs(f'{name}/{emotion} {emotions[emotion]}')

    count = 0
    for emotion,image_pixels,usage in tqdm(zip(df['emotion'], df['pixels'], df['usage'])):
        #pixels are separated by spaces
        image_string = image_pixels.split(' ') 
        image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
        img = Image.fromarray(image_data)
        count_string = str(count).zfill(6)

        path = ''
        if usage == 'Training':
            path = 'data/train/'
        elif usage == 'PublicTest':
            path = 'data/test-public/'
        elif usage == 'PrivateTest':
            path = 'data/test-private/'
        else:
            print("Exception!")
        
        img.save(os.path.join(path, f'{emotion} {emotions[emotion]}', f'{emotions[emotion]}-{count_string}.png')) 
        count += 1
    
    print(f'Training Data size: {len(df[df["usage"] == "Training"])}')
    print(f'Validation Data size: {len(df[df["usage"] == "PublicTest"])}')
    print(f'Test Data size: {len(df[df["usage"] == "PrivateTest"])}')
    return df

def get_datagenerator(dataset,preprocessing_func, batch_size = 128, img_size = (48, 48), img_color = 'grayscale', aug=False):
    if aug:
        datagen = ImageDataGenerator(
                            rescale=1./255,
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            preprocessing_function=preprocessing_func)
    else:
        datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocessing_func)

    return datagen.flow_from_directory(
            dataset,
            target_size=img_size,
            color_mode=img_color,
            shuffle=True,
            class_mode='categorical',
            batch_size=batch_size)

