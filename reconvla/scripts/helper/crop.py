import os
import sys
import torch
import argparse
from tqdm import tqdm
from multiprocessing import Pool, get_context, Process, set_start_method
import time
import traceback
from ultralytics import YOLO
from main_worker import process_folder

def init_worker(args):

    gpu_id, model_path, image_src_dir, log_folder = args
    
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = YOLO(model_path).half().to(device)
        
        return {
            "model": model,
            "image_src_dir": image_src_dir,
            "log_folder": log_folder,
            "gpu_id": gpu_id
        }
    except Exception as e:
        print(f"进程 {os.getpid()} 加载模型失败: {str(e)}")
        traceback.print_exc()
        return None

def run_task(args):

    task, worker_data = args
    
    if worker_data is None:
        return (task, False, "Worker initialization failed")
    
    try:
        model = worker_data["model"]
        image_src_dir = worker_data["image_src_dir"]
        log_folder = worker_data["log_folder"]
        gpu_id = worker_data["gpu_id"]
        
        task_name = os.path.basename(task)
        task_log_folder = os.path.join(log_folder, f"gpu{gpu_id}", task_name)
        os.makedirs(task_log_folder, exist_ok=True)
        
        start_time = time.time()
        process_folder(task, model, image_src_dir, task_log_folder)
        elapsed = time.time() - start_time
        
        return (task, True, f"Success in {elapsed:.2f}s")
        
    except Exception as e:
        tb = traceback.format_exc()
        error_log = os.path.join(log_folder, f"gpu{gpu_id}_errors.log")
        with open(error_log, "a") as f:
            f.write(f"[{time.ctime()}] Task {task} failed: {str(e)}\n{tb}\n")
        return (task, False, str(e))

def get_all_subfolders(root_folder):

    subfolders = []
    for split in ["training", "validation"]:
        split_path = os.path.join(root_folder, split)
        if os.path.exists(split_path):
            for name in sorted(os.listdir(split_path)):
                full_path = os.path.join(split_path, name)
                if os.path.isdir(full_path):
                    subfolders.append(full_path)
    return subfolders

def launch_processes(root_folder, model_path, image_src_dir, log_folder, gpus, max_concurrent_per_gpu=16):

    os.makedirs(log_folder, exist_ok=True)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    subfolders = get_all_subfolders(root_folder)
    total_tasks = len(subfolders)
    num_gpus = len(gpus)
    
    if total_tasks == 0:
        print("没有找到任何任务文件夹，请检查路径！")
        return
    
    print(f"总任务数: {total_tasks}, 使用GPU: {gpus}")
    
    tasks_per_gpu = []
    for i in range(num_gpus):
        tasks_per_gpu.append(subfolders[i::num_gpus])
    
    processes = []
    
    for idx, gpu_id in enumerate(gpus):
        gpu_tasks = tasks_per_gpu[idx]
        if not gpu_tasks:
            continue
            
        pool_size = min(len(gpu_tasks), max_concurrent_per_gpu)
        
        p = Process(
            target=gpu_worker,
            args=(gpu_id, gpu_tasks, model_path, image_src_dir, log_folder, pool_size)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

def gpu_worker(gpu_id, gpu_tasks, model_path, image_src_dir, log_folder, pool_size):
    ctx = get_context("spawn")
    start_time = time.time()
    
    init_args = (gpu_id, model_path, image_src_dir, log_folder)
    worker_data = init_worker(init_args)
    
    if worker_data is None:
        print(f"GPU {gpu_id} 初始化失败，无法处理任务")
        return
    
    task_args = [(task, worker_data) for task in gpu_tasks]
    
    success_count = 0
    failure_count = 0
    
    with ctx.Pool(processes=pool_size) as pool:
        results = pool.imap_unordered(run_task, task_args)
        
        pbar = tqdm(total=len(gpu_tasks), desc=f"GPU {gpu_id}")
        for result in results:
            task, success, msg = result
            if success:
                success_count += 1
            else:
                failure_count += 1
                pbar.write(f"任务失败 [{os.path.basename(task)}]: {msg}")
            pbar.update(1)
        pbar.close()
    
    del worker_data["model"]
    torch.cuda.empty_cache()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(gpu_tasks) if len(gpu_tasks) > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"GPU {gpu_id} 完成: {success_count} 成功, {failure_count} 失败")
    print(f"总时间: {total_time:.2f}秒 | 平均每任务: {avg_time:.2f}秒")
    print(f"吞吐量: {len(gpu_tasks)/total_time:.2f} 任务/秒")
    print(f"{'='*50}")

if __name__ == "__main__":
    set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser(description="多GPU并行处理任务")
    parser.add_argument("--root_folder", type=str, default="/share/user/iperror/data/calvin_dataset_ABC_D_0710/crop_img/task_ABC_D")
    parser.add_argument("--model_path", type=str, default="/data/user/wsong890/user68/ziyang/project_data/runs/detect/yolov8s_calvin4/weights/best.pt")
    parser.add_argument("--npz_src_dir", type=str, default="/share/user/iperror/data/task_ABC_D")
    parser.add_argument("--log_folder", type=str, default="/data/user/wsong890/user68/ziyang/project_data/log")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_concurrent_per_gpu", type=int, default=4, 
                        help="每个GPU上最大并发进程数")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    available_gpus = [i for i in range(torch.cuda.device_count())]
    if not set(gpus).issubset(set(available_gpus)):
        print(f"警告: 请求的GPU {gpus} 不匹配可用GPU {available_gpus}")
        gpus = [g for g in gpus if g in available_gpus]
        if not gpus:
            print("错误: 没有可用的请求GPU 程序退出")
            sys.exit(1)
    
    launch_processes(
        root_folder=args.root_folder,
        model_path=args.model_path,
        image_src_dir=args.npz_src_dir,
        log_folder=args.log_folder,
        gpus=gpus,
        max_concurrent_per_gpu=args.max_concurrent_per_gpu
    )
    os.system("exit")
