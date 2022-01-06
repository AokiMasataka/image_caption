import os
import json
import tqdm
import pickle
from torchvision.transforms import Resize
from torchvision.io import read_image, write_png
from torchvision.io.image import ImageReadMode


def label_preprocess(load_json):
    with open(load_json, 'r', encoding='UTF-8') as f:
        train_json = json.load(f)

    images = train_json['images']
    ants = train_json['annotations']

    train_list = [None for _ in range(ants.__len__())]
    bar = tqdm.tqdm(ants)
    for index, ant in enumerate(bar):
        image_id = ant['image_id']
        for image in images:
            if image['id'] == image_id:
                train_list[index] = {'file_name': image['file_name'], 'caption': ant['caption']}
                break
    
    return train_list


def write(export, obj):
    with open(export, 'wb') as f:
        pickle.dump(obj, f)


def main():
    train_list = label_preprocess('stair_captions/stair_captions_v1.2_train.json')
    valid_list = label_preprocess('stair_captions/stair_captions_v1.2_val.json')

    all_list = train_list + valid_list

    for i in all_list[:100]:
        print(i)

    write('stair_captions/captions.json', all_list)


if __name__ == '__main__':
    main()
