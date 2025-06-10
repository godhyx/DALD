import math

import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formatting import Collect, to_tensor
from mmcv.ops import MultiScaleDeformableAttention
from libs.utils.lane_utils import sample_lane


@PIPELINES.register_module
class CollectCLRNet(Collect):
    def __init__(
        self,
        keys=None,
        meta_keys=None,
        max_lanes=4,
        num_points=72,
        img_w=800,
        img_h=320,
        keypoint=0,
    ):
        self.keys = keys
        self.meta_keys = meta_keys
        self.max_lanes = max_lanes
        self.n_offsets = num_points
        self.n_strips = num_points - 1
        self.strip_size = img_h / self.n_strips
        self.offsets_ys = np.arange(img_h, -1, -self.strip_size)
        self.keypoint = keypoint
        self.img_w = img_w
        self.img_h = img_h

    def convert_targets(self, results):
        old_lanes = results["gt_points"]
        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 2, old_lanes)
        if self.keypoint:
            lanes = (np.ones((self.max_lanes, 2 + 1 + 2 + 2 + self.n_offsets), dtype=np.float32) * -1e5)
        else:
            lanes = (np.ones((self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32) * -1e5)
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image, theta_1, theta_2 = sample_lane(lane, self.offsets_ys, self.img_w, self.img_h, self.keypoint)
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:  # to calculate theta
                continue

            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = (1 - len(xs_outside_image) / self.n_strips)  # y0, relative
            lanes[lane_idx, 3] = xs_inside_image[0]  # x0, absolute
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            if theta_1 is None:
                thetas = []
                for i in range(1, len(xs_inside_image)):
                    theta = (
                        math.atan(
                            i
                            * self.strip_size
                            / (xs_inside_image[i] - xs_inside_image[0] + 1e-5)
                        )
                        / math.pi
                    )
                    theta = theta if theta > 0 else 1 - abs(theta)
                    thetas.append(theta)
                theta_far = sum(thetas) / len(thetas)
                lanes[lane_idx, 4] = theta_far  # theta
                lanes[lane_idx, 5] = len(xs_inside_image)  # length
                lanes[lane_idx, 6: 6 + len(all_xs)] = all_xs  # xs, absolute
            else:
                lanes[lane_idx, 4] = theta_1
                lanes[lane_idx, 5] = theta_2
                lanes[lane_idx, 6] = len(xs_inside_image)
                lanes[lane_idx, 7:7 + len(all_xs)] = all_xs




        results["lanes"] = to_tensor(lanes)
        return results

    def __call__(self, results):
        data = {}
        img_meta = {}
        if "lanes" in self.meta_keys:  # training
            results = self.convert_targets(results)
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data["img_metas"] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data
