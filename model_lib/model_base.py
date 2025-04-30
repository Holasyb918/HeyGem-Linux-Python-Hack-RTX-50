# -- coding: utf-8 --
# @Time : 2022/7/29



from .base_wrapper import ONNXModel
from pathlib import Path


try:
    from .base_wrapper import TRTWrapper, TRTWrapperSelf
except:
    pass


# from cv2box.utils import try_import

class ModelBase:
    def __init__(self, model_info, provider):
        self.model_path = model_info['model_path']

        if 'input_dynamic_shape' in model_info.keys():
            self.input_dynamic_shape = model_info['input_dynamic_shape']
        else:
            self.input_dynamic_shape = None

        if 'picklable' in model_info.keys():
            picklable = model_info['picklable']
        else:
            picklable = False

        if 'trt_wrapper_self' in model_info.keys():
            TRTWrapper = TRTWrapperSelf

        # init model
        if Path(self.model_path).suffix == '.engine':
            self.model_type = 'trt'
            self.model = TRTWrapper(self.model_path)
        elif Path(self.model_path).suffix == '.tjm':
            self.model_type = 'tjm'
            self.model = TJMWrapper(self.model_path, provider=provider)
        elif Path(self.model_path).suffix in ['.onnx', '.bin']:
            self.model_type = 'onnx'
            if not picklable:
                if 'encrypt' in model_info.keys():
                    self.model_path = load_encrypt_model(self.model_path, key=model_info['encrypt'])
                self.model = ONNXModel(self.model_path, provider=provider, input_dynamic_shape=self.input_dynamic_shape)
            else:
                self.model = OnnxModelPickable(self.model_path, provider=provider, )
        else:
            raise 'check model suffix , support engine/tjm/onnx now.'
