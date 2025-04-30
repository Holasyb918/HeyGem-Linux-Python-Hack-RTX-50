# HeyGem-Linux-Python-Hack-RTX-50
HeyGem for RTX50 series 


[![License](https://img.shields.io/badge/License-View%20License-blue.svg)](https://github.com/GuijiAI/HeyGem.ai/blob/main/LICENSE)
![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Linux](https://img.shields.io/badge/OS-Linux-brightgreen.svg)

**[中文](#chinese-version)** | **[English](README_en.md)**  

---
如果你需要较为完整的 HeyGem，即从 TTS 到数字人，那么你可以参考 [这里](README_tts_f2f.MD)
---

<a name="chinese-version"></a>

# HeyGem-Linux-Python-Hack-RTX-50
## 本项目支持 NVIDIA RTX 50 系列显卡，欢迎测试使用！

## 项目简介

[HeyGem-Linux-Python-Hack-RTX-50] 是一个基于 Python 的数字人项目，它从 [HeyGem.ai](https://github.com/GuijiAI/HeyGem.ai) 中提取出来，它能够直接在 Linux 系统上运行，摆脱了对 Docker 和 Windows 系统的依赖。我们的目标是提供一个更易于部署和使用的数字人解决方案。  
微信群请参考 https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/issues/37  

**如果你觉得这个项目对你有帮助，欢迎给我们 Star！**  
**如果运行过程中遇到问题，在查阅已有 Issue 后，在查阅 Google/baidu/ai 后，欢迎提交 Issues！**

## 主要特性

* 无需 Docker: 直接在 Linux 系统上运行，简化部署流程。
* 无需 Windows: 完全基于 Linux 开发和测试。
* Python 驱动: 使用 Python 语言开发，易于理解和扩展。
* 开发者友好: 易于使用和扩展。
* 完全离线。  

## 开始使用

### 安装
本项目**支持且仅支持 Linux & python3.9 环境**  
**AudoDL 可以参考** https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/issues/43  
请确保你的 Linux 系统上已经安装了 **Python 3.9**。然后，使用 pip 安装项目依赖项   
**具体的 onnxruntime-gpu / torch 等需要结合你的机器上的 cuda 版本去尝试一些组合，否则仍旧可能遇到问题。**  
**请尽量不要询问任何关于 pip 的问题，感谢合作**

```bash
# 直接安装整个 requirements.txt 不一定成功，更建议跑代码观察报错信息，然后根据报错信息结合 requirements 去尝试安装，祝你顺利。
# pip install -r requirements.txt
```

### 使用
把项目克隆到本地
```bash
git clone https://github.com/Holasyb918/HeyGem-Linux-Python-Hack-RTX-50
cd HeyGem-Linux-Python-Hack-RTX-50
bash download.sh
```
#### 开始使用  
* repo 中已提供可以用于 demo 的音视频样例，代码可以直接运行。  
#### command:  
```bash
python run.py 
```  

* 如果要使用自己的数据，可以外部传入参数，请注意，**path 是本地文件，且仅支持相对路径**.  

#### command:  
```bash
python run.py --audio_path example/audio.wav --video_path example/video.mp4
```  
#### gradio:  
```bash
python app.py
# 请等待模型初始化完成后提交任务
```

## QA
### 1. 多个人脸报错  
下载新的人脸检测模型，替换原本的人脸检测模型或许可以解决。
```bash
wget https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/scrfd_10g_kps.onnx
mv face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx.bak
mv scrfd_10g_kps.onnx face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx
```
### 2. 初始化报错  

有较高概率是 onnxruntime-gpu 版本不匹配导致的。  
```bash
python check_env/check_onnx_cuda.py
```
观察输出是否包括 successfully.  
如果遇到问题，你可以尝试以下方法：
1. 建议根据自己 cuda 等环境尝试更换一些版本。  
2. 如果难以解决，先卸载 onnxruntime-gpu 和 onnxruntime，然后使用 conda 安装 cudatoolkit 环境，然后再尝试 pip 安装 onnxruntime-gpu。    

    验证可行版本如下：  
    | cudatoolkit | onnxruntime-gpu | 备注 |
    | --- | --- | --- |
    | 11.8.0 | 1.16.0 |  |

### 3. ImportError: cannot import name check_argument_types  
缺包
```bash
pip install typeguard
```
  
### 4. library.so 找不到  
报错一般是类似于 Could not load library libcublasLt.so.11. Error: libcublasLt.so.11: cannot open shared object file: No such file or directory  

执行以下命令查看是否有改文件  
```
sudo find /usr -name "libcublasLt.so.11"  
```
没有的话，应该需要安装对应版本的cuda  
如果有的话就把第一步查看的文件路径添加到环境变量  
```
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
永久生效就添加到 ~/.bashrc 里面然后 source ~/.bashrc 一下  

## Contributing  
欢迎贡献！

## License
参考 heyGem.ai 的协议.
