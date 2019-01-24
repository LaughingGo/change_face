import json
ann_file = 'data/annotation_full.json'
image_dir = '../celebA/img_align_celeba/'
anns = json.load(open(ann_file, 'r'))
person_ids = anns.keys()
index_list = []
for k in range(1,1000):
    person_id = '{:5d}'.format(k)
    person_anns = anns[person_id]
    person_imgs_num = len(person_anns)
    print(person_imgs_num)
    for i in range(person_imgs_num):
        for j in range(person_imgs_num):
             if anns[person_id][i]['attribute'][0] == anns[person_id][j]['attribute'][0]:
                continue
            index_list.append({'person_id': person_id , 'img_1':i, 'img_2':j})
json.dump(index_list, open('data/select_data_full.json','w'))