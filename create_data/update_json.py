import json
import os
import numpy as np

def update_json():

    json_ann_train = json.load(open(os.path.join('./train_updated.json')))
    json_ann_val = json.load(open(os.path.join('./val_updated.json')))


    for ann_t in  json_ann_train['annotations'] :
        ann_t['image_id'] = ann_t['template_image_id']
        #ann_t['image_name'] = ann_t['template_image_name']
        #ann_t['bbox'] = ann_t['template_bbox']

    for ann_v in json_ann_val['annotations']:
        ann_v['image_id'] = ann_v['template_image_id']
        #ann_v['image_name'] = ann_v['template_image_name']
        #ann_v['bbox'] = ann_v['template_bbox']

    json_name = '%s.json'
    with open(os.path.join('./', json_name % 'train'), 'w') as outfile:
        json.dump(json_ann_train, outfile)

    with open(os.path.join('./', json_name % 'val'), 'w') as outfile:
        json.dump(json_ann_val, outfile)



if __name__ == '__main__':
    update_json()