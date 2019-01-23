import json
ann_file = 'data/annotation.json'
image_dir = '../celebA/img_align_celeba/'
anns = json.load(open(ann_file, 'r'))
person_ids = anns.keys()
index_list = []
for i in range(1,1000):
    person_id = '{:5d}'.format(person_id)
    person_anns = anns[person_id]
    person_imgs_num = len(person_anns)
    print(person_imgs_num)
    for i in range(person_imgs_num):
        for j in range(person_imgs_num):
            if j==i:
                continue
            index_list.append({'person_id': person_id , 'img_1':i, 'img_2':j})
json.dump(index_list, open('data/select_data.json','w'))