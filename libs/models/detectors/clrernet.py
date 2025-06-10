from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector


@DETECTORS.register_module()
class CLRerNet(SingleStageDetector):
    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        """CLRerNet detector."""
        super(CLRerNet, self).__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg
        )

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """
        Single-image test without augmentation.
        Args:
            img (torch.Tensor): Input image tensor of shape (1, 3, height, width).
            img_metas (List[dict]): Meta dict containing image information.
        Returns:
            result_dict (List[dict]): Single-image result containing prediction outputs and
             img_metas as 'result' and 'metas' respectively.
        """
        if "crop_offset" in img_metas[0]:
            self.bbox_head.test_cfg.cut_height = img_metas[0]["crop_offset"][1]
            self.bbox_head.test_cfg.ori_img_h = img_metas[0]["ori_shape"][0]
            self.bbox_head.test_cfg.ori_img_w = img_metas[0]["ori_shape"][1]
        x = self.extract_feat(img)
        output = self.bbox_head.simple_test(x)
        result_list = []
        for (out, meta) in zip(output, img_metas):
            meta["batch_input_shape"] = tuple(img.size()[-2:])
            result_dict = {
                "result": out,
                "meta": meta,
            }
            result_list.append(result_dict)
        return result_list  # assuming batch size is 1
