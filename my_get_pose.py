import argparse
import glob
import logging
import os
import sys
from typing import Any, ClassVar, Dict, List
import torch

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
# from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
# from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)

class InferenceAction():
    def __init__(self):
        self.cfg_path='configs/densepose_rcnn_R_50_FPN_s1x.yaml'
        self.model='/root/kj_work/IDM-VTON_old/local_directory/models--yisol--IDM-VTON/snapshots/585a32e74aee241cbc0d0cc3ab21392ca58c916a/densepose/model_final_162be9.pkl'
        self.opts = []
        self.opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        self.opts.append('0.8')
        self.cfg = get_cfg()
        add_densepose_config(self.cfg)
        self.cfg.merge_from_file(self.cfg_path)
        self.cfg.merge_from_list(self.opts)
        self.cfg.MODEL.WEIGHTS = self.model
        self.cfg.freeze()
        self.predictor = DefaultPredictor(self.cfg)

    def execute(self, input):
        # file_list = self._get_input_file_list(input)
        # if len(file_list) == 0:
        #     # logger.warning(f"No input images for {input}")
        #     return
        context = self.create_context()
        # for file_name in file_list:
        #     img = read_image(file_name, format="BGR")  # predictor expects BGR image.
        #     with torch.no_grad():
        #         outputs = self.predictor(img)["instances"]
        #         pose = self.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
        # return pose
        img = read_image(input, format="BGR")  # predictor expects BGR image.
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
            pose = self.execute_on_outputs(context, {"image": img}, outputs)
        return pose

    def _get_input_file_list(self, input_spec):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list

    def create_context(self) -> Dict[str, Any]:
        visualizers = []
        extractors = []

        texture_atlas = get_texture_atlas(None)
        texture_atlases_dict = get_texture_atlases(None)
        vis = DensePoseResultsFineSegmentationVisualizer(
            cfg=self.cfg,
            texture_atlas=texture_atlas,
            texture_atlases_dict=texture_atlases_dict,
        )
        visualizers.append(vis)
        extractor = create_extractor(vis)
        extractors.append(extractor)

        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": 'outputres.png',
            "entry_idx": 0,
        }
        return context
    
    def execute_on_outputs(
        self, context, entry, outputs
    ):
        import cv2
        import numpy as np

        visualizer = context["visualizer"]
        extractor = context["extractor"]
        # image_fpath = entry["file_name"]
        # logger.info(f"Processing {image_fpath}")
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        entry_idx = context["entry_idx"] + 1
        out_fname = self._get_out_fname(entry_idx, context["out_fname"])
        # out_dir = os.path.dirname(out_fname)
        # if len(out_dir) > 0 and not os.path.exists(out_dir):
        #     os.makedirs(out_dir)
        # cv2.imwrite(out_fname, image_vis)
        # logger.info(f"Output saved to {out_fname}")
        context["entry_idx"] += 1

        return image_vis

    def _get_out_fname(self, entry_idx, fname_base):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext

if __name__ == "__main__":
    a = InferenceAction()
    img = a.execute('/root/kj_work/IDM-VTON_old/my_tryon_test_data/img.jpg')