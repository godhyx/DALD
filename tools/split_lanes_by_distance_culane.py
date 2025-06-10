import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Split CULane lane annotations into near and far lanes.")
    parser.add_argument("root_path", type=str, help="CULane dataset root path (should contain 'list' folder).")
    parser.add_argument("new_path", type=str, help="Output path for split lane annotations.")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_splits = ['train.txt', 'val.txt', 'test.txt']
    list_paths = [os.path.join(args.root_path, 'list', fname) for fname in dataset_splits]
    path_1_4 = os.path.join(args.new_path, '1_4')   # y <= 350
    path_3_4 = os.path.join(args.new_path, '3_4')   # y >= 350

    for list_file in list_paths:
        with open(list_file) as f:
            for line in tqdm(f, desc=f"Processing {os.path.basename(list_file)}"):
                parts = line.strip().split()
                img_rel_path = parts[0].lstrip('/')
                img_path = os.path.join(args.root_path, img_rel_path)

                anno_path = img_path[:-3] + 'lines.txt'
                if not os.path.exists(anno_path):
                    continue

                with open(anno_path, 'r') as anno_file:
                    raw_data = [list(map(float, l.split())) for l in anno_file.readlines()]

                lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
                          if lane[i] >= 0 and lane[i + 1] >= 0] for lane in raw_data]
                lanes = [list(set(lane)) for lane in lanes if len(lane) > 2]
                lanes = [sorted(lane, key=lambda p: -p[1]) for lane in lanes]

                near_lanes, far_lanes = [], []
                for lane in lanes:
                    near = [(x, y) for x, y in lane if y <= 350]
                    far = [(x, y) for x, y in lane if y >= 350]

                    if len(near) >= 2:
                        near_lanes.append(' '.join(f"{x:.5f} {y:.5f}" for x, y in near))
                    if len(far) >= 2:
                        far_lanes.append(' '.join(f"{x:.5f} {y:.5f}" for x, y in far))

                # 保存文件
                file_parts = img_rel_path.split('/')
                save_dir_near = os.path.join(path_1_4, *file_parts[-3:-1])
                save_dir_far = os.path.join(path_3_4, *file_parts[-3:-1])
                os.makedirs(save_dir_near, exist_ok=True)
                os.makedirs(save_dir_far, exist_ok=True)

                save_path_near = os.path.join(path_1_4, img_rel_path[:-4] + '.lines.txt')
                save_path_far = os.path.join(path_3_4, img_rel_path[:-4] + '.lines.txt')

                with open(save_path_near, 'w') as f:
                    f.write('\n'.join(near_lanes))
                with open(save_path_far, 'w') as f:
                    f.write('\n'.join(far_lanes))

if __name__ == "__main__":
    main()