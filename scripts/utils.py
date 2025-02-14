import pandas as pd
import numpy as np
from PIL import Image
import os

TRAINING_CSV_PATH = '../../data/train/train.csv'
TEST_CSV_PATH = '../../data/test/test.csv'

TRAIN_IMAGES_PATH = '../../data/train/images/'
TEST_IMAGES_PATH = '../../data/test/images/'

id2label = {
    1 : "AI Generated",
    0 : "Human Generated"
}

def load_training_data():
    train_df = pd.read_csv(TRAINING_CSV_PATH, index_col='id')
        
    train_df['img_id'] = train_df['file_name'].apply(lambda x: x.split('/')[1])
    train_df['img_path'] = train_df['img_id'].apply(lambda x: os.path.join(TRAIN_IMAGES_PATH, x))
    train_df['pair_id'] = train_df.index // 2

    train_df.drop(columns=['file_name'], inplace=True)
    
    return train_df

def load_test_data():
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    test_df['img_id'] = test_df['id'].apply(lambda x: x.split('/')[1])
    test_df['img_path'] = test_df['img_id'].apply(lambda x: os.path.join(TEST_IMAGES_PATH, x))
    
    return test_df