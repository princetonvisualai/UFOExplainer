from __future__ import print_function, division
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms as T
import pickle


def create_ade20k_dataset():
    import csv
    import pandas as pd
    import os

    if os.path.exists('ADE20k/ade20k_imagelabels_with_texture.pkl'):
        A = pickle.load(open('ADE20k/ade20k_imagelabels_with_texture.pkl', 'rb'))
        images_train = A['train']
        images_val = A['val']
        images_test = A['test']
        labels = A['labels']
    else:
        image_df = pd.read_csv('dataset/broden1_224/index.csv')
        #image_df = image_df[image_df['split']==split]
        category_df = image_df[image_df['scene'].notnull()]
        print(category_df.head()) 
        category_maps = {}
        for cat in ['color','object','material', 'texture']:
            c_maps = pd.read_csv('dataset/broden1_224/c_{}.csv'.format(cat))
            cat_id_to_overall_id = {}
            for idx in c_maps.index:
                cat_id_to_overall_id[c_maps['number'][idx]] =c_maps['code'][idx] 
            category_maps[cat] = cat_id_to_overall_id 
        images = []
        labels = {}
        print(category_df['image'])
        for idx in category_df.index:
            #if (category_df['image'][idx]).split('/')[0]!='ade20k':
            #    continue
            full_image_name = 'dataset/broden1_224/images/{}'.format(category_df['image'][idx])
            print(idx)
            images.append(full_image_name)
            labels[full_image_name] = []

            for cat in ['color','object','material', 'texture']:
                #print(category_df[cat].notnull())
                #input()
                if category_df[cat].notnull()[idx]:
                    #print("hello", cat, idx, type(category_df[cat][idx]))
                    if isinstance(category_df[cat][idx], str):
                        img_labels = Image.open('dataset/broden1_224/images/{}'.format(category_df[cat][idx]))
                        numpy_val = np.array(img_labels)[:, :, 0]+ 256* np.array(img_labels)[:, :, 1]
                        code_val = [i for i in np.sort(np.unique(numpy_val))[1:]]
                        labels[full_image_name] += code_val
                    else:
                        labels[full_image_name].append(category_maps[cat][category_df[cat][idx]])
        from sklearn.model_selection import train_test_split

        images_train, images_valtest = train_test_split(images, test_size=0.4, random_state=42)
        images_val, images_test = train_test_split(images_valtest, test_size=0.5, random_state=42)
        
        with open('ADE20k/ade20k_imagelabels_with_texture.pkl', 'wb+') as handle:
            pickle.dump({'train': images_train, 'val':images_val, 'test':images_test, 'labels':labels}, handle)
    
                        

if __name__=='__main__':
    
    params = {'batch_size': 1,
             'shuffle': False,
             'num_workers': 0}
    
    create_ade20k_dataset()
    A = pickle.load(open('ade20k_imagelabels_with_texture.pkl', 'rb'))
    attr_list = []
    for key in A['labels']:
        attr_list+=A['labels'][key]

    attr_list = np.array(attr_list)
    print(np.unique(attr_list).shape)
