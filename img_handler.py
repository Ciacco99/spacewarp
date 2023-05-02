import shutil
import os
import json
import pandas as pd
import cv2 as cv

PATH_JPG = "Data/jpg/"
PATH_JSON = "Data/json/"

class Image:
    def __init__(self, path_json, path_jpg):
        self.path_img = path_json
        self.path_json = path_jpg

        self.img = cv.imread(path_jpg)

    def show_img(self):
        cv.namedWindow('image')
        cv.imshow('image', self.img)
    
        cv.waitKey(0)
        cv.destroyAllWindows()



class Image_handler:
    def __init__(self, path_folder_json, path_folder_jpg):
        self.path_json = path_folder_json        
        self.path_jpg = path_folder_jpg

        list_json = os.listdir(path_folder_json)
        list_jpg = os.listdir(path_folder_jpg)

        assert len(list_json)== len(list_jpg), "not the same number of file in json and jpg"

        list_files = []
        for (f_json, f_jpg) in zip(list_json, list_jpg):
            with open(path_folder_json + f_json) as f:
                contents = dict(json.load(f))

            img = Image(path_folder_json+f_json, path_folder_jpg + f_jpg)

            contents['Image'] = img

            list_files.append(contents)
        
        self.df = pd.DataFrame.from_dict(list_files)

    def head(self, size = None):
        print(self.df.head(size))

    def img_to_trash(self, key_img):

        img = self.df.loc[self.df['key'] == key_img]['Image'].values[0]
        
        shutil.()
        
        
        

        


img_H = Image_handler(PATH_JSON, PATH_JPG)
img_H.img_to_trash('000000001')


### HELPERS ###########

def merge_JsonFiles(filename, output_file):
    result = list()
    for f1 in filename:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    with open('images.json', 'w') as output_file:
        json.dump(result, output_file)

'''
PATH_JPG = "Data/jpg/"
PATH_JSON = "Data/json/"
CHOSEN = "000000001"

#image = Image(PATH_JPG+CHOSEN+".jpg", PATH_JSON+CHOSEN+".json")

img = Image("Data/jpg/000000001.jpg", "Data/json/000000001.json")

img.show_img()
'''

