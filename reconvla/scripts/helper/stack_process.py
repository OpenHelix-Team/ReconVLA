import os
import re
import cv2
import numpy as np
# from ultralytics import YOLO
from tqdm import tqdm
import pybullet as p
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import json
import argparse

# ==== 相机参数和投影相关函数 ====
fov = 10
aspect = 1
nearval = 0.01
farval = 10
width = 200
height = 200
look_at = [-0.026242351159453392, -0.0302329882979393, 0.3920000493526459]
look_from = [2.871459009488717, -2.166602199425597, 2.555159848480571]
up_vector = [0.4041403970338857, 0.22629790978217404, 0.8862616969685161]

viewMatrix = p.computeViewMatrix(cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=up_vector)
projectionMatrix = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=nearval, farVal=farval)

def project(point, viewMatrix, projectionMatrix, width, height):
    persp_m = np.array(projectionMatrix).reshape((4, 4)).T
    view_m = np.array(viewMatrix).reshape((4, 4)).T
    world_pix_tran = persp_m @ view_m @ point
    world_pix_tran = world_pix_tran / world_pix_tran[-1]
    world_pix_tran[:3] = (world_pix_tran[:3] + 1) / 2
    x, y = world_pix_tran[0] * width, (1 - world_pix_tran[1]) * height
    x, y = np.floor(x).astype(int), np.floor(y).astype(int)
    return (x, y)

def crop_center(img, u, v, crop_size=32):
    h, w = img.shape[:2]
    half = crop_size // 2
    left = max(u - half, 0)
    right = min(u + half, w)
    top = max(v - half, 0)
    bottom = min(v + half, h)
    crop_img = img[top:bottom, left:right, :]
    return crop_img

# ==== 目标类别映射 ====
id_map = {
    0: "button", 1: "blue block", 2: "red block", 3: "swich",
    4: "drawer", 5: "slider", 6: "pink block",
    7: "drawer inside", 8: "slider inside", 9: "gripper"
}
name2id = {v: k for k, v in id_map.items()}

block_names = ['red block', 'blue block', 'pink block']
block_indices = {
    'red block': (6, 9),
    'blue block': (12, 15),
    'pink block': (18, 21)
}

pattern = re.compile(r'frame_(\d+)\.png')


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (x2 - x1)*(y2 - y1) + (x2_p - x1_p)*(y2_p - y1_p) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def match_label_and_predict(image_path, label_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    gt_boxes = []
    with open(label_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i >= 2: break
            _, cx, cy, bw, bh = map(float, line.strip().split())
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            gt_boxes.append((x1, y1, x2, y2))
    results = model(image_path, conf=0, verbose=False)
    predictions = results[0].boxes
    pred_boxes = predictions.xyxy.cpu().numpy()
    pred_classes = predictions.cls.cpu().numpy()
    matched_classes = []
    for gt_box in gt_boxes:
        best_iou = 0
        best_class = -1
        for pred_box, cls in zip(pred_boxes, pred_classes):
            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_class = int(cls)
        matched_classes.append(id_map.get(best_class, "Unknown"))
    return matched_classes

def count_color_pixels(img, color):
    if color == 'red block' or color == 'red':
        mask = (img[..., 0] > 120) & (img[..., 1] < 130) & (img[..., 2] < 130)
    elif color == 'blue block' or color == 'blue':
        mask = (img[..., 2] > 120) & (img[..., 0] < 130) & (img[..., 1] < 130)
    elif color == 'pink block' or color == 'pink':
        mask = (img[..., 0] > 120) & (img[..., 2] > 120) & (img[..., 1] < 150)
    else:
        return 0
    return np.sum(mask)

def process_task_folder(task_path):
    import sys
    block_crop_indices = {
        'red block': (6, 9),
        'blue block': (12, 15),
        'pink block': (18, 21)
    }
    block_names_order = ['red block', 'blue block', 'pink block']


    print(f"Processing task folder: {task_path}")
    img_dir = os.path.join(task_path, 'img')
    crop_dir = os.path.join(task_path, 'crop')
    os.makedirs(crop_dir, exist_ok=True)
    annotations = []  # 用于存储当前task所有crop信息

    frame_ids = [int(pattern.match(f).group(1)) for f in os.listdir(img_dir) if pattern.match(f)]
    if not frame_ids:
        print("No frame images found, skip.")
        return
    max_frame_id = max(frame_ids) - 3 
    max_img_path = os.path.join(img_dir, f"frame_{max_frame_id:07d}.png")
    max_img_npz = os.path.join(npz_rootpath, 'training' if 'training' in task_path else 'validation', f"episode_{max_frame_id:07d}.npz")
    data = np.load(max_img_npz, allow_pickle=True)

    img = cv2.imread(max_img_path)
    if img is None:
        print(f"图片 {max_img_path} 读取失败")
        return

    block_centers = []
    uv_centers = []
    for name in block_names_order:
        start, end = block_crop_indices[name]
        center = data['scene_obs'][start:end]
        point = np.array([*center, 1])
        u, v = project(point, viewMatrix, projectionMatrix, width, height)
        uv_centers.append((u, v))
        block_centers.append(center)
    block_centers = np.array(block_centers)
    uv_centers = np.array(uv_centers)

    from itertools import combinations
    min_dist = float('inf')
    pair = None
    for (i, j) in combinations(range(len(uv_centers)), 2):
        dist = np.linalg.norm(uv_centers[i] - uv_centers[j])
        if dist < min_dist:
            min_dist = dist
            pair = (i, j)

    def crop_square(img, center, size=2):
        x, y = int(center[0]), int(center[1])
        half = size // 2
        x1, x2 = max(0, x - half), min(img.shape[1], x + half)
        y1, y2 = max(0, y - half), min(img.shape[0], y + half)
        return img[y1:y2, x1:x2]
    
    i,j = pair
    crop1 = crop_square(img, uv_centers[i])
    crop2 = crop_square(img, uv_centers[j])

    

    

    color1 = max({color: count_color_pixels(crop1, color) for color in block_names}.items(), key=lambda x: x[1])[0]
    color2 = max({color: count_color_pixels(crop2, color) for color in block_names}.items(), key=lambda x: x[1])[0]
    print(f"{task_path} Color1: {color1}, Color2: {color2}")
    # z轴高的是target_obj
    if block_centers[i][2] > block_centers[j][2]:
        target_obj, destination_obj = color1, color2
    else:
        target_obj, destination_obj = color2, color1
    
    print(f"Target object: {target_obj}, Destination object: {destination_obj}")



    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    for fname in tqdm(img_files, desc=f"Cropping images in {os.path.basename(task_path)}"):
        frame_id = int(pattern.match(fname).group(1))
        image_path = os.path.join(img_dir, fname)
        npz_path = os.path.join(npz_rootpath, 'training' if 'training' in task_path else 'validation', f"episode_{frame_id:07d}.npz")
        if not os.path.exists(npz_path):
            npz_path = npz_path.replace('training', 'validation')
            if not os.path.exists(npz_path):
                continue
        data = np.load(npz_path, allow_pickle=True)
        if 'rel_actions' not in data:
            continue
        rel_actions = data['rel_actions']
        if len(rel_actions) == 0:
            continue
        gripper_action = rel_actions[-1]
        if gripper_action == 1:
            chosen_obj = target_obj
        elif gripper_action == -1:
            chosen_obj = target_obj
        else:
            continue
        if chosen_obj not in block_names:
            continue
        # 用transfer.py的方式crop 3个block
        scene_obs = data['scene_obs']
        img = cv2.imread(image_path)
        block_crop_indices = {
            'red block': (6, 9),
            'blue block': (12, 15),
            'pink block': (18, 21)
        }
        block_names_order = ['red block', 'blue block', 'pink block']
        crops = []
        color_counts = []
        coords = []  # 记录每个crop对应的bbox坐标
        for name in block_names_order:
            start, end = block_crop_indices[name]
            xyz = scene_obs[start:end]
            point = np.array([*xyz, 1])
            u, v = project(point, viewMatrix, projectionMatrix, width, height)
            half = 32 // 2
            left = max(u - half, 0)
            right = min(u + half, img.shape[1])
            top = max(v - half, 0)
            bottom = min(v + half, img.shape[0])
            crop_img = img[top:bottom, left:right]
            crops.append(crop_img)
            coords.append({"x1": int(left), "y1": int(top), "x2": int(right), "y2": int(bottom)})
            color_counts.append(count_color_pixels(crop_img, chosen_obj))
        # 选择像素最多的crop，若都低于阈值则用原block的crop
        max_idx = int(np.argmax(color_counts))
        threshold = 5
        if color_counts[max_idx] < threshold:
            # 阈值不满足，回退用对应block的crop
            fallback_idx = block_names_order.index(chosen_obj)
            crop = crops[fallback_idx]
            coord = coords[fallback_idx]
        else:
            crop = crops[max_idx]
            coord = coords[max_idx]
        # 保存crop为jpg
        crop_fname = fname.replace('.png', '.jpg')
        save_path = os.path.join(crop_dir, crop_fname)
        cv2.imwrite(save_path, crop)
        # 记录annotation
        annotations.append({
            "npz": os.path.basename(npz_path),
            "original_image": fname,
            "crop_image": crop_fname,
            "label": "stack block",
            "coordinates": coord
        })

    # 将annotations写入json文件
    if annotations:
        json_path = os.path.join(crop_dir, 'crop_info.jsonl')
        with open(json_path, "w", encoding="utf-8") as jf:
            for anno in annotations:
                json.dump(anno, jf, ensure_ascii=False)
                jf.write("\n")

def main():
    cpu_count = os.cpu_count() or 1  # 动态分配CPU数
    for split in ['training', 'validation']:
        split_path = os.path.join(root_path, split)
        if not os.path.exists(split_path):
            continue
        # 只保留文件夹名中带有_stack_block的
        task_folders = [
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d)) and '_stack_block' in d
        ]
        task_paths = [os.path.join(split_path, task_name) for task_name in task_folders]
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = {executor.submit(process_task_folder, task_path): task_path for task_path in task_paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split} tasks (parallel)"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="多GPU并行处理任务")
    parser.add_argument("--log_path", type=str, default="./scripts/process_colorblock.log")
    # ==== 路径设置 ====
    parser.add_argument("--root_folder", type=str, default="/share/user/iperror/data/calvin_dataset_ABC_D_0710/crop_img/task_ABC_D")
    parser.add_argument("--npz_src_dir", type=str, default="/share/user/iperror/data/task_ABC_D")
    args = parser.parse_args()

    log_path = args.log_path
    root_path = args.root_folder
    npz_rootpath = args.npz_src_dir
    
    def redirect_stdout_stderr(log_path):
        log_file = open(log_path, "w", buffering=1, encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file

    redirect_stdout_stderr(log_path)
    main() 
