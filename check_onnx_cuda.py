import onnxruntime
import numpy as np

def check_gpu_usage():
    """
    Checks if ONNX Runtime can use the GPU by attempting to create an InferenceSession
    with the CUDAExecutionProvider.

    Returns:
        True if GPU is likely being used, False otherwise.
    """
    providers = ("CUDAExecutionProvider",
             {"device_id": 0})
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 3
    onnx_path = "./face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx"
    onnx_session = onnxruntime.InferenceSession(onnx_path, session_options, providers=[providers])
    print(onnx_session.get_providers())
    return "CUDAExecutionProvider" in onnx_session.get_providers(), onnx_session

if __name__ == "__main__":
    is_cuda, onnx_session = check_gpu_usage()
    if is_cuda:
        print("ONNX Runtime is successfully using the GPU.")
        inp = np.random.randn(1, 3, 640, 640).astype(np.float32)
        ort_inputs = {onnx_session.get_inputs()[0].name: inp}
        ort_outs = onnx_session.run(None, ort_inputs)
        print(ort_outs[0].shape)
    else:
        print("ONNX Runtime is NOT using the GPU or there was an error initializing the CUDA provider.")
        print("Please ensure that:")
        print("- You have installed the 'onnxruntime-gpu' package.")
        print("- You have a compatible NVIDIA GPU with appropriate drivers installed.")
        print("- CUDA and cuDNN are installed and correctly configured in your system.")
        print("- The versions of CUDA, cuDNN, and the NVIDIA drivers are compatible with the 'onnxruntime-gpu' version you have installed.")
        print("- The ONNX Runtime build you are using supports CUDA.")
        