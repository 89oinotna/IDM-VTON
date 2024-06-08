import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import time

def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 3).astype(np.float32) +
                    (parse_array == 11).astype(np.float32))
    parse_lower = ((parse_array == 6).astype(np.float32) +
                    (parse_array == 5).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))
                    # (parse_array == 18).astype(np.float32) +
                    # (parse_array == 19).astype(np.float32))
    # print(img.size)
    # print(parse.shape)
    w, h = parse.shape[1], parse.shape[0]
    # img = img.resize((w,h))
    img = Image.new("RGB", (w,h), (0, 0, 0))
    # print(img.size)
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    # print(pose_data[5],pose_data[2],np.linalg.norm(pose_data[5] - pose_data[2]))
    # print(pose_data[12],pose_data[9],np.linalg.norm(pose_data[12] - pose_data[9]))
    # exit()
    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[11] - pose_data[8])
    point = (pose_data[8] + pose_data[11]) / 2
    pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a
    r = int(length_a / 16) + 1
    
    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'white', width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'white', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white')

    # mask torso
    for i in [8, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'white', 'white')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 8]], 'white', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], 'white', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], 'white', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 11, 8]], 'white', 'white')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'white', 'white')
    # print(agnostic.split())
    # print(img.split())
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic

def get_img_agnostic2(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_shoe = ((parse_array == 9).astype(np.float32) +
                    (parse_array == 10).astype(np.float32))
    parse_upper = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 14).astype(np.float32) +
                    (parse_array == 15).astype(np.float32))
    
    parse_belt = ((parse_array == 8).astype(np.float32))
    
    # print(img.size)
    print(parse.shape)
    w, h = parse.shape[1], parse.shape[0]
    # img = img.resize((w,h))
    img = Image.new("RGB", (w,h), (0, 0, 0))
    img2 = Image.new("RGB", (w,h), (255, 255, 255))
    # print(img.size)
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)
    

    # print(pose_data[5],pose_data[2],np.linalg.norm(pose_data[5] - pose_data[2]))
    # print(pose_data[12],pose_data[9],np.linalg.norm(pose_data[12] - pose_data[9]))
    # exit()
    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[11] - pose_data[8])
    point = (pose_data[8] + pose_data[11]) / 2
    pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a
    r = int(length_a / 16) + 1
    
    # mask leg
    agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], 'white', width=r*10)
    # agnostic.save('/root/kj_work/idm_output/mask.jpg')
    # time.sleep(5)
    for i in [8, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white')
        # agnostic.save('/root/kj_work/idm_output/mask.jpg')
        # time.sleep(5)
    for i in [9, 10, 12, 13]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'white', width=r*10)
        # agnostic.save('/root/kj_work/idm_output/mask.jpg')
        # time.sleep(5)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white')
        # agnostic.save('/root/kj_work/idm_output/mask.jpg')
        # time.sleep(5)

    # mask torso
    for i in [10, 13]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'white', 'white')
        # agnostic.save('/root/kj_work/idm_output/mask.jpg')
        # time.sleep(5)

    agnostic_draw.line([tuple(pose_data[i]) for i in [8, 10]], 'white', width=r*6)
    # agnostic.save('/root/kj_work/idm_output/mask.jpg')
    # time.sleep(5)

    agnostic_draw.line([tuple(pose_data[i]) for i in [11, 13]], 'white', width=r*6)
    # agnostic.save('/root/kj_work/idm_output/mask.jpg')
    # time.sleep(5)

    agnostic_draw.line([tuple(pose_data[i]) for i in [10, 13]], 'white', width=r*12)
    # agnostic.save('/root/kj_work/idm_output/mask.jpg')
    # time.sleep(5)

    agnostic_draw.polygon([tuple(pose_data[i]) for i in [8, 11, 13, 10]], 'white', 'white')
    # agnostic.save('/root/kj_work/idm_output/mask.jpg')
    # time.sleep(5)

    # mask neck
    pointx, pointy = pose_data[10]
    agnostic_draw.rectangle((pointx-r*6, pointy-r*6, pointx+r*6, pointy+r*6), 'white', 'white')
    # agnostic.save('/root/kj_work/idm_output/mask.jpg')
    # time.sleep(5)
    pointx, pointy = pose_data[13]
    agnostic_draw.rectangle((pointx-r*6, pointy-r*6, pointx+r*6, pointy+r*6), 'white', 'white')
    # agnostic.save('/root/kj_work/idm_output/mask.jpg')
    # time.sleep(5)

    # print(agnostic.split())
    # print(img.split())
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_shoe * 255), 'L'))
    # agnostic.save('/root/kj_work/idm_output/mask.jpg')
    # time.sleep(5)
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(img2, None, Image.fromarray(np.uint8(parse_belt * 255), 'L'))
    
    
    return agnostic

def get_img_agnostic3(img, parse, pose_data):
    w, h = parse.shape[1], parse.shape[0]
    img = Image.new("RGB", (w,h), (255, 255, 255))
    agnostic_u = get_img_agnostic(img.copy(), parse, pose_data).convert('L')
    agnostic_l = get_img_agnostic2(img.copy(), parse, pose_data).convert('L')
    # agnostic_l.paste(img, None, Image.fromarray(agnostic_u, 'L'))
    combined_image = Image.fromarray((np.array(agnostic_u) | np.array(agnostic_l)).astype(np.uint8), 'L')

    return combined_image

if __name__ =="__main__":
    data_path = './test'
    output_path = './test/parse'
    
    os.makedirs(output_path, exist_ok=True)
    
    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        
        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        
        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
        except IndexError:
            print(pose_name)
            continue

        # load parsing image
        im = Image.open(osp.join(data_path, 'image', im_name))
        label_name = im_name.replace('.jpg', '.png')
        im_label = Image.open(osp.join(data_path, 'image-parse-v3', label_name))

        agnostic = get_img_agnostic(im, im_label, pose_data)
        
        agnostic.save(osp.join(output_path, im_name))