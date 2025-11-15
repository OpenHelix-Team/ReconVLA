import tensorflow as tf
import json
import os
import glob
import random
from typing import Dict, Any, Generator
from decimal import Decimal
import numpy as np
from PIL import Image
import io
import argparse
import base64
from tqdm import tqdm

TARGET_IMG_SIZE = 334

def convert_decimals(obj):
    """
    递归地遍历一个对象，将所有Decimal类型转换为float类型。
    """
    if isinstance(obj, list):
        return [convert_decimals(i) for i in obj]
    if isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


class Bridge_PretrainDataLoader:


    def __init__(self, tfrecord_dir: str, json_path: str, image_save_dir: str):
        """
        初始化数据加载器。
        Args:
            tfrecord_dir (str): TFRecord文件的目录。
            json_path (str): JSON文件的路径。
            image_save_dir (str): 保存完整图像和裁剪图像的目录。 (此功能将被注释)
        """
        self.tfrecord_dir = tfrecord_dir
        self.json_path = json_path


        self.tfrecord_files = glob.glob(os.path.join(self.tfrecord_dir, "*.tfrecord*"))
        if not self.tfrecord_files:
            raise FileNotFoundError(f"在目录中未找到.tfrecord文件: {self.tfrecord_dir}")
        print(f"找到 {len(self.tfrecord_files)} 个TFRecord文件。")

        print("正在加载Bridge JSON文件...")
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
            print("JSON文件加载完毕。")
        except Exception as e:
            raise e

    def _choose_random_action(self) -> str:
        common_actions = ["pick up", "place", "push", "pull", "open", "close", "grasp", "reach for", "move", "lift", "put"]
        return random.choice(common_actions)

    def _find_target_box(self, bboxes: list, subtask: str, gripper_point: list | None) -> list:
        if not bboxes: return None
        stop_words = {'a', 'an', 'the'}
        best_match = None
        if subtask:
            subtask_words = set(subtask.lower().split()) - stop_words
            max_confidence = -1
            for box in bboxes:
                try:
                    confidence, label, coords = box
                    if isinstance(label, str):
                        label_words = set(label.lower().split()) - stop_words
                        if subtask_words.intersection(label_words) and confidence > max_confidence:
                            max_confidence = confidence
                            best_match = box
                except (ValueError, TypeError): continue
        if best_match is None:
            # 若提供 gripper_point，则生成以其为中心的 55x55 方框
            if gripper_point and len(gripper_point) >= 2:
                x_c, y_c = gripper_point[0], gripper_point[1]  # x,y
                half = 37  # 55x55 --> 半边 27
                y_min = max(0, int(y_c - half))
                x_min = max(0, int(x_c - half))
                y_max = int(y_c + half)
                x_max = int(x_c + half)

                # 找到与 gripper_point 最近的原始 bbox 的 label
                nearest_label = 'gripper_box'
                nearest_dist = float('inf')
                for b in bboxes:
                    try:
                        _, lbl, c = b
                        cx = (c[1] + c[3]) / 2  # x_center
                        cy = (c[0] + c[2]) / 2  # y_center
                        dist = (cx - x_c) ** 2 + (cy - y_c) ** 2
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_label = lbl if isinstance(lbl, str) else 'gripper_box'
                    except Exception:
                        continue

                return [1.0, nearest_label, [y_min, x_min, y_max, x_max]]
            # 否则随机返回任一框
            return random.choice(bboxes)
        return best_match

    def get_pretrain_data(self, debug_mode: bool = False) -> Generator[Dict[str, Any], None, None]:
        for tf_file in self.tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tf_file)
            for raw_record in raw_dataset:
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature

                    file_path_key = features['episode_metadata/file_path'].bytes_list.value[0].decode('utf-8')
                    episode_id_key = str(features['episode_metadata/episode_id'].int64_list.value[0])
                    
                    json_episode_data = self.json_data.get(file_path_key, {}).get(episode_id_key)
                    if not json_episode_data:
                        if debug_mode: print(f"DEBUG: 跳过 episode {episode_id_key}，因为在JSON中未找到。")
                        continue

                    image_bytes_list = features['steps/observation/image_0'].bytes_list.value
                    image_bytes_list_wrist = features['steps/observation/image_3'].bytes_list.value
                    action_bytes_list = features['steps/action'].float_list.value
                    action_group = []
                    for i in range(0, len(action_bytes_list), 7):
                        action_group.append(action_bytes_list[i:i+7])
                    num_steps = len(image_bytes_list)
                    

                    json_features = json_episode_data.get('features', {})
                    bboxes_list = json_features.get('bboxes', [])
                    gripper_points_list = json_features.get('gripper_position', [])
                    json_reasoning = json_episode_data.get('reasoning', {})
                    
                    subtasks_list = []
                    if isinstance(json_reasoning, dict):
                        for i in range(num_steps):
                            step_reasoning = json_reasoning.get(str(i), {})
                            subtasks_list.append(step_reasoning.get('subtask', ''))
                    
                    # 7 elements per action
                    if not (num_steps == len(bboxes_list) and num_steps == len(subtasks_list) ):
                        if debug_mode: 
                            print(f"DEBUG: 跳过 episode {episode_id_key}，因为数据长度不匹配。")
                            print(f"  - 图像步骤: {num_steps}")
                            print(f"  - BBoxes 步骤: {len(bboxes_list)}")
                            print(f"  - Subtasks 步骤: {len(subtasks_list)}")
                        continue

                    for i in range(num_steps):
                        image_bytes = image_bytes_list[i]
                        image_bytes_wrist = image_bytes_list_wrist[i] if i < len(image_bytes_list_wrist) else None
                        current_bboxes = bboxes_list[i]
                        current_subtask = subtasks_list[i]
                        current_action = action_group[i]

                        if not current_bboxes:
                            if debug_mode: print(f"DEBUG: 跳过 step {i} of episode {episode_id_key}，因为没有 bboxes。")
                            continue

                        # 取 gripper point
                        gripper_point = None
                        if i < len(gripper_points_list):
                            # 形如 [[x,y]] 或 [x,y]
                            gp = gripper_points_list[i]
                            if isinstance(gp, list) and len(gp) == 1:
                                gp = gp[0]
                            gripper_point = gp if isinstance(gp, list) else None

                        target_box = self._find_target_box(current_bboxes, current_subtask, gripper_point)
                        
                        if not current_subtask and target_box:
                            act_str = self._choose_random_action()
                            label = target_box[1]
                            if isinstance(label, str) and label:
                                current_subtask = f"{act_str} {label}"
                        
                        target_image_bytes = None
                        if target_box:
                            try:
                                image_tensor = tf.image.decode_image(image_bytes, channels=3)
                                h, w, _ = image_tensor.shape
                                coords = target_box[2]
                                y_min, x_min, y_max, x_max = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                                h_off, w_off = max(0, y_min), max(0, x_min)
                                h_t, w_t = max(0, min(h, y_max) - h_off), max(0, min(w, x_max) - w_off)

                                if h_t > 0 and w_t > 0:
                                    crop_tensor = tf.image.crop_to_bounding_box(image_tensor, h_off, w_off, h_t, w_t)
                                    target_image_bytes = tf.image.encode_jpeg(crop_tensor).numpy()
                            except Exception:
                                target_image_bytes = None

                        # ---------- 将 top & wrist 图像拼接为一张 (vertical) ----------

                        def prep_img(bytes_data, default_black=False):
                            if bytes_data is None:
                                if default_black:
                                    return tf.zeros((TARGET_IMG_SIZE//2, TARGET_IMG_SIZE, 3), dtype=tf.uint8)
                                else:
                                    return None
                            img = tf.image.decode_image(bytes_data, channels=3)
                            img = tf.image.resize(img, [TARGET_IMG_SIZE//2, TARGET_IMG_SIZE], method="bilinear")
                            img = tf.cast(img, tf.uint8)
                            return img

                        top_tensor = prep_img(image_bytes, default_black=True)
                        wrist_tensor = prep_img(image_bytes_wrist, default_black=True)

                        combined_tensor = tf.concat([top_tensor, wrist_tensor], axis=0)  # vertical stack
                        combined_bytes = tf.io.encode_jpeg(combined_tensor).numpy()

                        # target image prep (top crop + duplicate on bottom)
                        target_top_tensor = None
                        if target_image_bytes:
                            target_top_tensor = prep_img(target_image_bytes, default_black=False)
                        else:
                            target_top_tensor = tf.zeros((TARGET_IMG_SIZE//2, TARGET_IMG_SIZE, 3), dtype=tf.uint8)

                        target_combined_tensor = tf.concat([target_top_tensor, wrist_tensor], axis=0)  # 上: target, 下: wrist
                        target_combined_bytes = tf.io.encode_jpeg(target_combined_tensor).numpy()

                        # --- 构建最终输出格式 ---
                        sample_id = f"bridge_ep_{episode_id_key}_step_{i}"
                        
                        image_b64 = base64.b64encode(combined_bytes).decode('utf-8')
                        target_image_b64 = base64.b64encode(target_combined_bytes).decode('utf-8') if target_combined_bytes else None

                        # 若有关键字段缺失则跳过本 sample
                        if (not image_b64) or (not target_image_b64) or (not current_subtask):
                            if debug_mode:
                                print(f"DEBUG: 跳过 sample {sample_id}，关键字段缺失。subtask:{bool(current_subtask)} image:{bool(image_b64)} target:{bool(target_image_b64)}")
                            continue
                        
                        
                        action_str = " ".join(map(str, current_action))
                        yield {
                            'id': sample_id,
                            'image': image_b64,
                            'image_target': target_image_b64,
                            'conversations': [
                                {'from': 'human', 'value': f"{current_subtask}\n<image>\n{current_subtask}"},
                                {'from': 'gpt', 'value': f"{action_str}"}
                            ],
                            'embody': False
                        }
                except Exception:
                    continue
                
                if debug_mode:
                    print("--- 调试模式: 已处理完一个episode，停止加载。 ---")
                    return


class Libero_PretrainDataLoader:


    def __init__(self, tfrecord_dir: str, json_path: str, image_save_dir: str):
        self.tfrecord_dir = tfrecord_dir
        self.json_path = json_path


        self.tfrecord_files = glob.glob(os.path.join(self.tfrecord_dir, "*.tfrecord*"))
        if not self.tfrecord_files:
            raise FileNotFoundError(f"在目录中未找到.tfrecord文件: {self.tfrecord_dir}")
        print(f"找到 {len(self.tfrecord_files)} 个TFRecord文件。")

        print("正在加载Libero JSON文件...")
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
            print("JSON文件加载完毕。")
        except Exception as e:
            raise e

    def _get_box_center(self, box: list) -> tuple:
        try:
            y0, x0 = box[0]; y1, x1 = box[1]
            return ((x0 + x1) / 2, (y0 + y1) / 2)
        except (IndexError, TypeError): return None

    def _find_target_box_label(self, json_episode_data: dict, num_steps: int) -> str:
        if num_steps < 2: return None
        first_step = json_episode_data.get('0', {}); last_step = json_episode_data.get(str(num_steps - 1), {})
        first_bboxes = first_step.get('bboxes', {}); last_bboxes = last_step.get('bboxes', {})
        if not first_bboxes or not last_bboxes: return None

        common_labels = set(first_bboxes.keys()) & set(last_bboxes.keys())
        max_displacement, target_label = -1, None
        for label in common_labels:
            if 'gripper' in label.lower(): continue
            center_first = self._get_box_center(first_bboxes.get(label))
            center_last = self._get_box_center(last_bboxes.get(label))
            if center_first and center_last:
                displacement = np.linalg.norm(np.array(center_last) - np.array(center_first))
                if displacement > max_displacement:
                    max_displacement, target_label = displacement, label
        return target_label

    def get_pretrain_data(self, debug_mode: bool = False) -> Generator[Dict[str, Any], None, None]:
        for tf_file in self.tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tf_file)
            for raw_record in raw_dataset:
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature

                    file_path_key = features['episode_metadata/file_path'].bytes_list.value[0].decode('utf-8')
                    demo_id_key = str(features['episode_metadata/demo_id'].int64_list.value[0])
                    
                    json_episode_data = self.json_data.get(file_path_key, {}).get(demo_id_key)
                    if not json_episode_data:
                        if debug_mode: print(f"DEBUG: 跳过 demo {demo_id_key}，因为在JSON中未找到。")
                        continue

                    image_bytes_list = features['steps/observation/image'].bytes_list.value
                    image_bytes_list_wrist = features['steps/observation/wrist_image'].bytes_list.value
                    actions = features['steps/action'].float_list.value
                    num_steps = len(image_bytes_list)
                    
                    target_box_label = self._find_target_box_label(json_episode_data, num_steps)
                    if not target_box_label:
                        if debug_mode: print(f"DEBUG: 跳过 demo {demo_id_key}，因为无法找到目标物体标签。")
                        continue
                    
                    # Libero has 7 action dims
                    if num_steps * 7 != len(actions):
                        if debug_mode: print(f"DEBUG: 跳过 demo {demo_id_key}，因为action长度不匹配。")
                        continue

                    for i in range(num_steps):
                        step_json_data = json_episode_data.get(str(i))
                        if not step_json_data:
                            if debug_mode: print(f"DEBUG: 跳过 step {i} of demo {demo_id_key}，因为没有step json数据。")
                            continue
                        
                        image_bytes = image_bytes_list[i]
                        image_bytes_wrist = image_bytes_list_wrist[i] if i < len(image_bytes_list_wrist) else None
                        action = actions[i*7 : (i+1)*7]
                        current_bboxes = step_json_data.get('bboxes', {})
                        target_box_coords = current_bboxes.get(target_box_label)
                        
                        target_image_bytes = None
                        if target_box_coords:
                            try:
                                original_image = Image.open(io.BytesIO(image_bytes))
                                y0, x0 = target_box_coords[0]
                                y1, x1 = target_box_coords[1]
                                cropped_image = original_image.crop((x0, y0, x1, y1))
                                
                                buf = io.BytesIO()
                                cropped_image.save(buf, format='PNG')
                                target_image_bytes = buf.getvalue()
                            except Exception:
                                target_image_bytes = None

                        # ---------- 拼接顶视 & 腕部视图 ----------

                        def prep_img(bytes_data, default_black=False):
                            if bytes_data is None:
                                if default_black:
                                    return tf.zeros((TARGET_IMG_SIZE//2, TARGET_IMG_SIZE, 3), dtype=tf.uint8)
                                else:
                                    return None
                            img = tf.image.decode_image(bytes_data, channels=3)
                            img = tf.image.resize(img, [TARGET_IMG_SIZE//2, TARGET_IMG_SIZE], method="bilinear")
                            img = tf.cast(img, tf.uint8)
                            return img

                        top_tensor = prep_img(image_bytes, default_black=True)
                        wrist_tensor = prep_img(image_bytes_wrist, default_black=True)

                        combined_tensor = tf.concat([top_tensor, wrist_tensor], axis=0)
                        combined_bytes = tf.io.encode_jpeg(combined_tensor).numpy()

                        # target 上+腕下
                        if target_image_bytes:
                            target_top_tensor = prep_img(target_image_bytes, default_black=False)
                        else:
                            target_top_tensor = tf.zeros((TARGET_IMG_SIZE//2, TARGET_IMG_SIZE, 3), dtype=tf.uint8)

                        target_combined_tensor = tf.concat([target_top_tensor, wrist_tensor], axis=0)
                        target_combined_bytes = tf.io.encode_jpeg(target_combined_tensor).numpy()

                        # --- 构建最终输出格式 ---
                        sample_id = f"libero_ep_{demo_id_key}_step_{i}"

                        image_b64 = base64.b64encode(combined_bytes).decode('utf-8')
                        target_image_b64 = base64.b64encode(target_combined_bytes).decode('utf-8') if target_combined_bytes else None
                        action_str = " ".join(map(str, action))
                        yield {
                            'id': sample_id,
                            'image': image_b64,
                            'image_target': target_image_b64,
                            'conversations': [
                                {'from': 'human', 'value': f"{step_json_data.get('subtask', '')}\n<image>\n{step_json_data.get('subtask', '')}"},
                                {'from': 'gpt', 'value': f"{action_str}"}
                            ],
                            'embody': False
                        }
                except Exception:
                    continue

                if debug_mode:
                    print("--- 调试模式: 已处理完一个episode，停止加载。 ---")
                    return


if __name__ == '__main__':
    # Example usage
    parser = argparse.ArgumentParser(description="Run a specified dataset loader for testing.")
    parser.add_argument('--dataset', type=str, required=True, choices=['bridge', 'libero'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_json', type=str, default='./json/ecot/pretrain/pretrain_vla.json')
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    debug_save_path = None
    if args.debug:
        debug_save_path = f"./debug/debug_images/debug_images_{args.dataset}"
        os.makedirs(debug_save_path, exist_ok=True)
        print(f"--- 调试模式已开启，恢复的图像将保存在: {debug_save_path} ---")

    if args.dataset == 'bridge':
        print("--- Testing Bridge Dataloader ---")
        loader = Bridge_PretrainDataLoader(
            tfrecord_dir="/data/embodied-CoT/data/bridge_orig/1.0.0",
            json_path="/data/embodied-CoT/data/bridge_orig/embodied_features_bridge.json",
            image_save_dir="./debug/bridge_pretrain_vla"
        )
    else:
        print("--- Testing Libero Dataloader ---")
        loader = Libero_PretrainDataLoader(
            tfrecord_dir="/data/embodied-CoT/data/embodied_features_and_demos_libero/libero_lm_90/1.0.0",
            json_path="/data/embodied-CoT/data/embodied_features_and_demos_libero/libero_reasonings.json",
            image_save_dir="./debug/libero_pretrain_vla"
        )

    generator = loader.get_pretrain_data(debug_mode=args.debug)
    
    if args.output_json:
        print(f"--- 正在生成样本并写入到 {args.output_json} ---")
        all_samples = []
        # 使用tqdm包装generator，并添加描述
        data_generator = tqdm(generator, desc=f"正在处理 {args.dataset} 样本")
        
        for i, sample in enumerate(data_generator):
            all_samples.append(sample)
            # 如果指定了样本数量，则在达到数量后停止
            if args.num_samples and i + 1 >= args.num_samples:
                print(f"\n已达到指定的样本数量 {args.num_samples}，停止处理。")
                break
        
        print(f"\n正在将 {len(all_samples)} 个样本写入JSON文件，这可能需要一些时间...")
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        print(f"\n--- 完成！总共 {len(all_samples)} 个样本已成功写入到 {args.output_json} ---")
    else:
        print(f"\n--- {args.dataset.capitalize()} Pre-training Samples (Debug/Display Mode) ---")
        for i, sample in enumerate(generator):
            if i >= 5 and not args.debug: break # 在非debug模式下最多显示5个
            print(f"\n--- Sample {i+1} ---")
            # 打印部分字段以验证格式
            print(f"  ID: {sample['id']}")
            print(f"  Image (Base64 length): {len(sample['image'])}")
            print(f"  Image Target (Base64 length): {len(sample['image_target']) if sample['image_target'] else 0}")
            print(f"  Conversation (Human): {sample['conversations'][0]['value']}")
            print(f"  Embody: {sample['embody']}")

            if args.debug:
                try:
                    # 解码并保存原图
                    img_bytes = base64.b64decode(sample['image'])
                    img_filename = os.path.join(debug_save_path, f"{sample['id']}.jpg")
                    with open(img_filename, 'wb') as f:
                        f.write(img_bytes)
                    print(f"    -> 已保存恢复的原图到: {img_filename}")

                    # 解码并保存目标图
                    if sample['image_target']:
                        target_img_bytes = base64.b64decode(sample['image_target'])
                        target_filename = os.path.join(debug_save_path, f"{sample['id']}_target.jpg")
                        with open(target_filename, 'wb') as f:
                            f.write(target_img_bytes)
                        print(f"    -> 已保存恢复的目标图到: {target_filename}")
                except Exception as e:
                    print(f"    -> 图像恢复失败: {e}") 
