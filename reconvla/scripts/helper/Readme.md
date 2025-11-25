### For CALVIN
```bash
cd reconvla/reconvla
python ./scripts/helper/crop.py \
--root_folder /output/path/ \
--model_path ./scripts/helper/best.pt \
--npz_src_dir  /path/to/training/ \
```
Below is an explanation of the parameters：
- `npz_src_dir`: Source directory containing episode NPZ files.
- `root_folder`: Output root folder for extracted task.

#### For 'Stack' task
```bash
cd reconvla/reconvla
python ./scripts/helper/stack_process.py \
--root_folder /output/path/ \
--npz_src_dir  /path/to/training/ \
``` 
#### For 'Unctack' task
Due to the absence of any annotations in CALVIN's text instructions for the 'unstack' task that could facilitate automatic extraction, and considering that the initial frame of this task always shows the object already grasped, only the target object needs to be annotated. Here are two proposed methods:

1. Manually save the last frame of each UNSTACK task and use tools like LabelImg to annotate the **bbox** of the target object, saving it in **XML** format. Then apply the following script to extract the target image.
```
output_folder/
├── 56_unstack_block/
│   ├── lang_ann/
│   └── img/
│   └── crop/
│   └── *.xml/  
```
```bash
cd reconvla/reconvla
python ./scripts/helper/unstack_process.py
--root_folder /output/path/ \
```

2. Read the object coordinate data from the last frame of unstack tasks in the lang_ann.yaml file, identify the two blocks with the closest geometric distance, select the block with the smaller z-axis value, and crop a fixed-size region as the target image.
