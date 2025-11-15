import os
import shutil
import cv2
import yaml
import json
import re
import numpy as np
from datetime import datetime

def extract_frame_number(file_name):
    match = re.search(r'frame_(\d+)\.png', file_name)
    return int(match.group(1)) if match else float('inf')

def filter_highest_confidence_boxes(results):
    boxes = results.boxes
    all_boxes = boxes.xyxy.cpu().numpy()
    all_confs = boxes.conf.cpu().numpy()
    all_classes = boxes.cls.cpu().numpy().astype(int)

    selected_boxes = []
    selected_confs = []
    selected_classes = []

    unique_classes = set(all_classes)
    for cls in unique_classes:
        idxs = [i for i, c in enumerate(all_classes) if c == cls]
        best_idx = idxs[0]
        best_conf = all_confs[best_idx]
        for i in idxs[1:]:
            if all_confs[i] > best_conf:
                best_conf = all_confs[i]
                best_idx = i
        selected_boxes.append(all_boxes[best_idx])
        selected_confs.append(best_conf)
        selected_classes.append(cls)

    return selected_boxes, selected_confs, selected_classes

def get_gripper_state(npz_path):
    ##print(npz_path)
    data = np.load(npz_path)
    actions = data['actions']
    gripper_state = actions[-1]
    return gripper_state

def select_obj(task_text , gripper_state):
    id_map = {
        0: "button",
        1: "blue block",
        2: "red block",
        3: "swich",
        4: "drawer",
        5: "slider",
        6: "pink block",
        7: "drawer inside",
        8: "slider inside",
        9: "gripper"
    }

    task_yolo_label_map_open = {
            "rotate_red_block_right": 2,
            "rotate_blue_block_right": 1,
            "rotate_pink_block_right": 6,
            "rotate_red_block_left": 2,
            "rotate_blue_block_left": 1,
            "rotate_pink_block_left": 6,

            "push_red_block_right": 2,
            "push_blue_block_right": 1,
            "push_pink_block_right": 6,
            "push_red_block_left": 2,
            "push_blue_block_left": 1,
            "push_pink_block_left": 6,

            "move_slider_left": 5,
            "move_slider_right": 5,

            "open_drawer": 4,
            "close_drawer": 4,

            #从A到B的需要重新设计！
            "lift_red_block_table": 2,
            "lift_blue_block_table": 1,
            "lift_pink_block_table": 6,

            "lift_red_block_slider": 2,
            "lift_blue_block_slider": 1,
            "lift_pink_block_slider": 6,

            "lift_red_block_drawer": 2,
            "lift_blue_block_drawer": 1,
            "lift_pink_block_drawer": 6,

            #goal是slider内部
            "place_in_slider": 8,
            "place_in_drawer": 7,

            # 99 代表要查找离gripper最近的block
            "push_into_drawer": 99,

            "turn_on_lightbulb": 3,
            "turn_off_lightbulb": 3,

            "turn_on_led": 0,
            "turn_off_led": 0
        }

    task_yolo_label_map_close = {
            "rotate_red_block_right": 2,
            "rotate_blue_block_right": 1,
            "rotate_pink_block_right": 6,
            "rotate_red_block_left": 2,
            "rotate_blue_block_left": 1,
            "rotate_pink_block_left": 6,

            "push_red_block_right": 2,
            "push_blue_block_right": 1,
            "push_pink_block_right": 6,
            "push_red_block_left": 2,
            "push_blue_block_left": 1,
            "push_pink_block_left": 6,

            "move_slider_left": 5,
            "move_slider_right": 5,

            "open_drawer": 4,
            "close_drawer": 4,

            #从A到B的需要重新设计！
            "lift_red_block_table": 2,
            "lift_blue_block_table": 1,
            "lift_pink_block_table": 6,

            "lift_red_block_slider": 2,
            "lift_blue_block_slider": 1,
            "lift_pink_block_slider": 6,

            "lift_red_block_drawer": 2,
            "lift_blue_block_drawer": 1,
            "lift_pink_block_drawer": 6,

            #goal是slider内部
            "place_in_slider": 8,
            "place_in_drawer": 7,

            "push_into_drawer": 7,

            "turn_on_lightbulb": 3,
            "turn_off_lightbulb": 3,

            "turn_on_led": 0,
            "turn_off_led": 0
        }
    #gripper_action (1): binary (close = -1, open = 1)
    if gripper_state == 1:
        return task_yolo_label_map_open.get(task_text, None)
    if gripper_state == -1:
        return task_yolo_label_map_close.get(task_text, None)

def process_folder(root_folder, model, image_src_dir, log_file_path):
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    batch_size=16

    now = datetime.now().strftime("%H:%M:%S")
    #print(f'----------subfolder_name :{root_folder}-----{now}------',flush=True)
    if not os.path.isdir(root_folder):
        raise FileNotFoundError(f"指定的 root_folder 不存在或不是目录: {root_folder}")

    img_folder = os.path.join(root_folder, "img")
    file_list = sorted(os.listdir(img_folder), key=extract_frame_number, reverse=True)
    lang_ann_path = os.path.join(root_folder, "lang_ann", "lang_ann.yaml")

    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"指定的 {root_folder}，因为没有 img 文件夹")

    with open(lang_ann_path, 'r') as f:
        ann_data = yaml.safe_load(f)

    #output_folder = os.path.join(root_folder, "output")
    crop_folder = os.path.join(root_folder, "crop")

    if os.path.exists(crop_folder):
        shutil.rmtree(crop_folder)
    #if os.path.exists(output_folder):
    #    shutil.rmtree(output_folder)
    os.makedirs(crop_folder, exist_ok=True)
    #os.makedirs(output_folder, exist_ok=True)

    jsonl_path = os.path.join(crop_folder, "crop_info.jsonl")
    is_first_image = True  
    is_flag_99_first = False

    batch_imgs = []
    batch_file_paths = []
    batch_file_names = []
    batch_npz_paths = []
    batch_task_texts = []
    batch_gripper_states = []
    batch_target_classes = []


    for file_name in file_list:
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in IMG_EXTENSIONS:
            raise ValueError(f"不支持的文件类型: {ext}")

        img_path = os.path.join(img_folder, file_name)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片：{img_path}")

        task_text = ann_data.get('task', None)
        if task_text is None:
            raise ValueError(f"{file_name} 无任务标签，跳过")
        
        file_num = file_name.split('_')[1].split('.')[0]
        file_num_int = int(file_num)
        npz_filename = f"episode_{file_num_int:07d}.npz"
        last_part = os.path.basename(os.path.dirname(root_folder))
        npz_path = os.path.join(image_src_dir, last_part, npz_filename)            
        gripper_state = get_gripper_state(npz_path)
        target_cls = select_obj(task_text, gripper_state)
        
        if target_cls is None:
            raise ValueError(f"{file_name} 任务 '{task_text}' 未映射到类别，跳过")

        batch_imgs.append(img)
        batch_file_paths.append(img_path)
        batch_file_names.append(file_name)
        batch_npz_paths.append(npz_filename)
        batch_task_texts.append(task_text)
        batch_gripper_states.append(gripper_state)
        batch_target_classes.append(target_cls)

        if len(batch_imgs) == batch_size:
            _process_batch(
                batch_imgs, batch_file_paths,batch_file_names, batch_npz_paths, batch_task_texts,
                batch_gripper_states, batch_target_classes, model,
                crop_folder, jsonl_path, log_file_path, is_first_image, is_flag_99_first
            )
            is_first_image = False
            is_flag_99_first = False
            batch_imgs.clear()
            batch_file_paths.clear()
            batch_file_names.clear()
            batch_npz_paths.clear()
            batch_task_texts.clear()
            batch_gripper_states.clear()
            batch_target_classes.clear()

    # 剩余的图片少于 batch_size，也处理掉
    if batch_imgs:
        _process_batch(
            batch_imgs, batch_file_paths,batch_file_names, batch_npz_paths, batch_task_texts,
            batch_gripper_states, batch_target_classes, model,
            crop_folder, jsonl_path, log_file_path, is_first_image, is_flag_99_first
        )


def _process_batch(imgs, batch_file_paths, file_names, npz_names, task_texts, gripper_states, target_classes,
                   model, crop_folder, jsonl_path, log_file_path, is_first_image, is_flag_99_first):    
    results_list = model(
        imgs, 
        conf=0,    
        verbose=False,     
        stream=False,      
        show=False         
    )

    for i, results in enumerate(results_list):
        img = imgs[i]
        file_name = file_names[i]
        file_path = batch_file_paths[i]
        npz_filename = npz_names[i]
        task_text = task_texts[i]
        gripper_state = gripper_states[i]
        target_cls = target_classes[i]

        boxes, confs, classes = filter_highest_confidence_boxes(results)

        h, w = img.shape[:2]

        if target_cls != 99:
            found_target = False
            for box, conf, cls in zip(boxes, confs, classes):
                if cls != target_cls:
                    continue  # 只保留任务对应类别
                found_target = True

                x1, y1, x2, y2 = box.astype(int)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                label = model.names[cls]
                text = f"{label} {conf:.2f}"
                # 这里可以做可视化，暂时不做

                crop_img = img[y1:y2, x1:x2]
                crop_name = f"{os.path.splitext(file_name)[0]}.jpg"
                crop_path = os.path.join(crop_folder, crop_name)

                if crop_img is None or crop_img.size == 0 or y1 >= y2 or x1 >= x2 or y2 > h or x2 > w:
                    cv2.imwrite(crop_path, img)
                else:
                    cv2.imwrite(crop_path, crop_img)

                crop_info = {
                    "npz": npz_filename,  
                    "original_image": file_name,
                    "crop_image": crop_name,
                    "label": label,
                    "coordinates": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                }
                mode = 'w' if is_first_image else 'a'
                with open(jsonl_path, mode) as jsonl_file:
                    jsonl_file.write(json.dumps(crop_info) + '\n')
                is_first_image = False

            if not found_target:
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"{crop_folder}/{file_name}\n")

        else:
            target9 = [(box, conf, cls) for box, conf, cls in zip(boxes, confs, classes) if cls == 9]
            if not target9:
                raise ValueError("没有找到 cls==9 的框，跳过该图像。")
            box9, _, _ = target9[0]
            cx9, cy9 = (box9[0] + box9[2]) / 2, (box9[1] + box9[3]) / 2

            parent_dir = os.path.dirname(file_path)
            frame_files = [f for f in os.listdir(parent_dir) if f.startswith('frame_') and f.endswith('.png')]
            frame_files.sort(key=extract_frame_number, reverse=True)
            target_file = os.path.join(parent_dir, frame_files[0])

            if not target_file:
                raise FileNotFoundError("在父目录中没有找到 frame_XXX.png 文件")
            if os.path.basename(target_file) == os.path.basename(file_path):
                candidate_boxes = [(box, conf, cls) for box, conf, cls in zip(boxes, confs, classes) if cls in (1, 2, 6)]
                if not candidate_boxes:
                    raise ValueError("没有找到 cls==1/2/6 的候选框，跳过该图像。")

                min_dist = float('inf')
                target_box, target_conf, target_cls_actual = None, None, None
                for box, conf, cls in candidate_boxes:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    dist = (cx - cx9)**2 + (cy - cy9)**2
                    if dist < min_dist:
                        min_dist = dist
                        target_box, target_conf, target_cls_actual = box, conf, cls


                for box, conf, cls in zip(boxes, confs, classes):
                    if cls != target_cls_actual:
                        continue

                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                    label = model.names[cls]
                    crop_img = img[y1:y2, x1:x2]
                    crop_name = f"{os.path.splitext(file_name)[0]}.jpg"
                    crop_path = os.path.join(crop_folder, crop_name)

                    if crop_img is not None and crop_img.size > 0:
                        cv2.imwrite(crop_path, crop_img)
                    else:
                        cv2.imwrite(crop_path, img)

                    crop_info = {
                        "npz": npz_filename,  
                        "original_image": file_name,
                        "crop_image": crop_name,
                        "label": label,
                        "coordinates": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        }
                    }
                    mode = 'w' if is_first_image else 'a'
                    with open(jsonl_path, mode) as jsonl_file:
                        jsonl_file.write(json.dumps(crop_info) + '\n')
                    is_first_image = False
        
            else:
                    results = model(
                                    cv2.imread(target_file), 
                                    conf=0,    
                                    verbose=False,     
                                    stream=False,      
                                    show=False         
                                )[0]
                    boxes, confs, classes = filter_highest_confidence_boxes(results)

                    target9 = [(box, conf, cls) for box, conf, cls in zip(boxes, confs, classes) if cls == 9]
                    if not target9:
                        #print(f"没有找到 cls==9 的框，跳过该图像：{file_name}")
                        raise ValueError("没有找到 cls==9 的框，跳过该图像。")
                    box9, _, _ = target9[0]
                    cx9, cy9 = (box9[0] + box9[2]) / 2, (box9[1] + box9[3]) / 2

                    candidate_boxes = [(box, conf, cls) for box, conf, cls in zip(boxes, confs, classes) if cls in (1, 2, 6)]
                    if not candidate_boxes:
                        #print(f"没有找到 cls==1/2/6 的候选框，跳过该图像：{file_name}")
                        raise ValueError("没有找到 cls==1/2/6 的候选框，跳过该图像。")
                    min_dist = float('inf')
                    target_box, target_conf, target_cls_actual = None, None, None
                    for box, conf, cls in candidate_boxes:
                        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        dist = (cx - cx9)**2 + (cy - cy9)**2
                        if dist < min_dist:
                            min_dist = dist
                            target_box, target_conf, target_cls_actual = box, conf, cls

                    img_annotated = img.copy()
                    h, w = img.shape[:2]
                    jsonl_path = os.path.join(crop_folder, "crop_info.jsonl")

                    for box, conf, cls in zip(boxes, confs, classes):
                        if cls != target_cls_actual:
                            continue  # 只保留任务对应类别

                        x1, y1, x2, y2 = box.astype(int)
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                        label = model.names[cls]
                        text = f"{label} {conf:.2f}"
                        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_annotated, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        crop_img = img[y1:y2, x1:x2]
                        crop_name = f"{os.path.splitext(file_name)[0]}.jpg"
                        crop_path = os.path.join(crop_folder, crop_name)
                        if crop_img is not None and crop_img.size > 0:
                            cv2.imwrite(crop_path, crop_img)
                        else:
                            #print(f"[警告] 无效裁剪图像，使用原图代替: {crop_path}")
                            cv2.imwrite(crop_path, img)

                        #save_path = os.path.join(output_folder, file_name)
                        #cv2.imwrite(save_path, img_annotated)


                        crop_info = {
                            "npz": npz_filename,  
                            "original_image": file_name,
                            "crop_image": crop_name,
                            "label": label,
                            "coordinates": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2)
                            }
                        }
                        mode = 'w' if is_first_image else 'a'
                        with open(jsonl_path, mode) as jsonl_file:
                            jsonl_file.write(json.dumps(crop_info) + '\n')
                        is_first_image = False
