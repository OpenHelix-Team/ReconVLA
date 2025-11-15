import os
import cv2
import xml.etree.ElementTree as ET
import sys
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_bounding_box(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bndbox = root.find(".//bndbox")
    if bndbox is None:
        return None
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)
    return (xmin, ymin, xmax, ymax)

def crop_all_images_in_folder(img_folder, crop_folder, box):
    os.makedirs(crop_folder, exist_ok=True)
    for file in sorted(os.listdir(img_folder)):
        if file.endswith(".png"):
            img_path = os.path.join(img_folder, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue
            xmin, ymin, xmax, ymax = box
            cropped = img[ymin:ymax, xmin:xmax]
            save_path = os.path.join(crop_folder, file)
            cv2.imwrite(save_path, cropped)
            print(f"✅ 裁剪并保存: {save_path}")

def process_one_subfolder(subfolder_path):
    img_dir = os.path.join(subfolder_path, "img")
    if not os.path.exists(img_dir):
        return f"❌ {subfolder_path} 没有img目录，跳过"
    xml_files = [f for f in os.listdir(subfolder_path) if f.endswith(".xml")]
    if not xml_files:
        return f"⚠️ {os.path.basename(subfolder_path)} 没有 XML 文件，跳过"
    xml_path = os.path.join(subfolder_path, xml_files[0])
    box = parse_bounding_box(xml_path)
    if box is None:
        return f"❌ XML 中未找到 bounding box: {xml_path}"
    crop_folder = os.path.join(subfolder_path, "crop")
    print(f"\n📂 正在处理: {os.path.basename(subfolder_path)}")
    print(f"🔍 裁剪区域: {box}")
    crop_all_images_in_folder(img_dir, crop_folder, box)
    return f"✅ 完成: {os.path.basename(subfolder_path)}"

def process_all_subfolders(base_root, splits, max_workers=None):
    total_count = 0
    subfolder_paths = []
    for split in splits:
        split_dir = os.path.join(base_root, split)
        for subfolder in os.listdir(split_dir):
            if "unstack" not in subfolder:
                continue
            subfolder_path = os.path.join(split_dir, subfolder)
            if os.path.isdir(subfolder_path):
                subfolder_paths.append(subfolder_path)

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    print(f"🚦 启动并行处理，进程数: {max_workers}")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_subfolder, path) for path in subfolder_paths]
        for future in as_completed(futures):
            result = future.result()
            print(result)
            if result.startswith("✅ 完成"):
                total_count += 1

    print(f"\n🚀 总共处理了 {total_count} 个子目录。")

if __name__ == "__main__":
    splits = ["training", "validation"]
    log_path = "log_task_ABC_D.txt"
    # 支持命令行传参调整进程数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_workers', type=int, default=None, help='并行进程数，默认自动检测')
    parser.add_argument("--root_folder", type=str, default="/share/user/iperror/data/calvin_dataset_ABC_D_0710/crop_img/task_ABC_D")

    args = parser.parse_args()
    max_workers = args.max_workers
    base_root = args.root_folder
    with open(log_path, "w") as log_file:
        with redirect_stdout(log_file):
            process_all_subfolders(base_root, splits, max_workers=max_workers)