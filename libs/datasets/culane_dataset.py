"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/culane_dataset.py
"""


import shutil
from pathlib import Path
import os
import cv2
import numpy as np
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.utils import get_root_logger
from tqdm import tqdm

from libs.datasets.metrics.culane_metric import eval_predictions
from libs.datasets.pipelines import Compose


@DATASETS.register_module
class CulaneDataset(CustomDataset):
    """Culane Dataset class."""

    def __init__(
        self,
        data_root,
        data_list,
        pipeline,
        diff_file=None,
        diff_thr=15,
        test_mode=True,
        y_step=2,
        ori_w=1640,
        ori_h=590,
    ):
        """
        Args:
            data_root (str): Dataset root path.
            data_list (str): Dataset list file path.
            pipeline (List[mmcv.utils.config.ConfigDict]):
                Data transformation pipeline configs.
            test_mode (bool): Test flag.
            y_step (int): Row interval (in the original image's y scale) to sample
                the predicted lanes for evaluation.

        """
        self.data_root = data_root
        if isinstance(data_root, list):
            self.img_prefix = data_root[0]
        else:
            self.img_prefix = data_root
        self.test_mode = test_mode
        self.ori_w, self.ori_h = ori_w, ori_h
        # read image list
        self.diffs = np.load(diff_file)["data"] if diff_file is not None else []
        self.diff_thr = diff_thr
        if isinstance(data_list, list):
            self.img_infos, self.annotations, self.mask_paths = self.parse_datalist(data_list[0])
        else:
            self.img_infos, self.annotations, self.mask_paths = self.parse_datalist(data_list)
        print(len(self.img_infos), "data are loaded")
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # build data pipeline
        self.pipeline = Compose(pipeline)
        self.result_dir = "tmp"
        self.list_path = data_list
        if test_mode:
            self.test_categories_dir = [str(Path(data).joinpath("list/test_split/")) for data in data_root]
        self.y_step = y_step

    def parse_datalist(self, data_list):
        """
        Read image data list.
        Args:
            data_list (str): Data list file path.
        Returns:
            List[str]: List of image paths.
        """
        img_infos, annotations, mask_paths = [], [], []
        with open(data_list) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if len(self.diffs) > 0 and self.diffs[i] < self.diff_thr:
                    continue
                img_paths = line.strip().split(" ")
                img_infos.append(img_paths[0].lstrip("/"))
                if not self.test_mode:
                    anno_path = img_paths[0].replace(".jpg", ".lines.txt")
                    annotations.append(anno_path.lstrip("/"))
                    if len(img_paths) > 1:
                        mask_paths.append(img_paths[1].lstrip("/"))
                    else:
                        mask_paths.append(None)
        return img_infos, annotations, mask_paths

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def prepare_train_img(self, idx):
        """
        Read and process the image through the transform pipeline for training.
        Args:
            idx (int): Data index.
        Returns:
            dict: Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        imgname = str(Path(self.img_prefix).joinpath(self.img_infos[idx]))
        sub_img_name = self.img_infos[idx]
        img = cv2.imread(imgname)
        ori_shape = img.shape
        kps, id_classes, id_instances = self.load_labels(idx)
        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=kps,
            id_classes=id_classes,
            id_instances=id_instances,
            img_shape=ori_shape,
            ori_shape=ori_shape,
            eval_shape=(
                320,
                1640,
            ),
            gt_masks=None,
        )
        if self.mask_paths[0]:
            results["gt_masks"] = self.load_mask(idx)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """
        Read and process the image through the transform pipeline for test.
        Args:
            idx (int): Data index.
        Returns:
            dict: Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        img_name = str(Path(self.img_prefix).joinpath(self.img_infos[idx]))
        sub_img_name = self.img_infos[idx]
        img = cv2.imread(img_name)
        ori_shape = img.shape
        results = dict(
            filename=img_name,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=[],
            id_classes=[],
            id_instances=[],
            img_shape=ori_shape,
            ori_shape=ori_shape,
        )
        return self.pipeline(results)

    def load_mask(self, idx):
        """
        Read a segmentation mask for training.
        Args:
            idx (int): Data index.
        Returns:
            numpy.ndarray: segmentation mask.
        """
        maskname = str(Path(self.img_prefix).joinpath(self.mask_paths[idx]))
        mask = cv2.imread(maskname, cv2.IMREAD_UNCHANGED)
        return mask

    def load_labels(self, idx, offset_y=0):
        """
        Read a ground-truth lane from an annotation file.
        Args:
            idx (int): Data index.
        Returns:
            List[list]: list of lane point lists.
            list: class id (=1) for lane instances.
            list: instance id (start from 1) for lane instances.
        """
        anno_dir = str(Path(self.img_prefix).joinpath(self.annotations[idx]))
        shapes = []
        with open(anno_dir, "r") as anno_f:
            lines = anno_f.readlines()
            for line in lines:
                coords = []
                coords_str = line.strip().split(" ")
                for i in range(len(coords_str) // 2):
                    coord_x = float(coords_str[2 * i])
                    coord_y = float(coords_str[2 * i + 1]) + offset_y
                    coords.append(coord_x)
                    coords.append(coord_y)
                if len(coords) > 3:
                    shapes.append(coords)
        id_classes = [1 for i in range(len(shapes))]
        id_instances = [i + 1 for i in range(len(shapes))]
        return shapes, id_classes, id_instances

    def evaluate(self, results, metric="F1", logger=None):
        """
        Write prediction to txt files for evaluation and
        evaluate them with labels.
        Args:
            results (List[dict]): All inference results containing:
                result (dict): contains 'lanes' and 'scores'.
                meta (dict): contains meta information.
            metric (str): Metric type to evaluate. (not used)
        Returns:
            dict: Evaluation result dict containing
                F1, precision, recall, etc. on the specified IoU thresholds.

        """
        all_dict = {}
        save_path = ['whole', 'near', 'far']
        for result in tqdm(results):
            lanes = result["result"]["lanes"]
            outputs = self.get_prediction_string(lanes)
            # input_image_path = result['meta']['filename']
            # output_image_path = 'vis/' + result['meta']['sub_img_name']
            # self.visualize_lane_lines(outputs[0], input_image_path, output_image_path)
            for i, (out, path_pre) in enumerate(zip(outputs, save_path)):
                tmp_save = self.result_dir + f"/{path_pre}"
                dst_path = (Path(tmp_save).joinpath(result["meta"]["sub_img_name"]).with_suffix(".lines.txt"))
                dst_path.parents[0].mkdir(parents=True, exist_ok=True)
                with open(str(dst_path), "w") as f:
                    output = "\n".join(out) if len(out) > 0 else ""
                    if len(output) > 0:
                        print(output, file=f)
        for i in range(3):
            eval_dict = eval_predictions(
                self.result_dir + f"/{save_path[i]}",
                self.data_root[i],
                self.list_path[i],
                self.test_categories_dir[i],
                logger=get_root_logger(log_level="INFO"),
            )
            new_dict = {key + f"{path_pre}": value for key, value in eval_dict.items()}
            all_dict.update(new_dict)

        shutil.rmtree(self.result_dir)
        return all_dict

    def visualize_lane_lines(self, outputs, input_image_path, output_image_path=None):
        
        image = cv2.imread(input_image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {input_image_path}")

       
        for line in outputs:
            points = list(map(float, line.split()))  
            points = np.array(points).reshape(-1, 2).astype(int)  
            for i in range(len(points) - 1):
                cv2.line(image, tuple(points[i]), tuple(points[i + 1]), color=(0, 255, 0), thickness=2)  

       
        if output_image_path:
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, image)

    def get_prediction_string(self, lanes):
        """
        Convert lane instance structure to prediction strings.
        Args:
            lanes (List[Lane]): List of lane instances in `Lane` structure.
        Returns:
            out_string (str): Output string.
        """
        ys = np.arange(0, self.ori_h, self.y_step) / self.ori_h
        out = []
        out_far = []
        out_near = []
        for lane in lanes:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.ori_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.ori_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            if len(lane_xs) < 2:
                continue
            lane_str = " ".join(["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != "":
                out.append(lane_str)
            lane_str = " ".join(["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys) if y >=350])
            if lane_str != "":
                out_near.append(lane_str)
            lane_str = " ".join(["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys) if y <= 350])
            if lane_str != "":
                out_far.append(lane_str)
        return [out, out_near, out_far]
        # return "\n".join(out) if len(out) > 0 else ""
