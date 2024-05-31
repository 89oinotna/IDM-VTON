import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference
import torch


class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.add_session_config_entry('gpu_id', str(gpu_id))
        self.session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), '../IDM-VTON_old/local_directory/models--yisol--IDM-VTON/snapshots/585a32e74aee241cbc0d0cc3ab21392ca58c916a/humanparsing/parsing_atr.onnx'),
                                            sess_options=session_options, providers=['CPUExecutionProvider'])
        self.lip_session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), '../IDM-VTON_old/local_directory/models--yisol--IDM-VTON/snapshots/585a32e74aee241cbc0d0cc3ab21392ca58c916a/humanparsing/parsing_lip.onnx'),
                                                sess_options=session_options, providers=['CPUExecutionProvider'])
        

    def __call__(self, input_image):
        # torch.cuda.set_device(self.gpu_id)
        # parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        # return parsed_image, face_mask
        parsed = onnx_inference(self.session, self.lip_session, input_image)
        return parsed

if __name__ == "__main__":
    p = Parsing(0)
    img, mask,parsing_result = p('/root/kj_work/IDM-VTON/my_pre_data/img')
    img.save('/root/kj_work/IDM-VTON/my_pre_data/pall.png')
    # mask.save('/root/kj_work/IDM-VTON/my_pre_data/p2.png')