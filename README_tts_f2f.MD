
[![License](https://img.shields.io/badge/License-View%20License-blue.svg)](https://github.com/GuijiAI/HeyGem.ai/blob/main/LICENSE)
![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![Linux](https://img.shields.io/badge/OS-Linux-brightgreen.svg)

**[中文](#chinese-version)** | **[English](README_en.md)**

---

<a name="chinese-version"></a>

# HeyGem-Linux-Python-Hack

## 项目简介

[HeyGem-Linux-Python-Hack] 是一个基于 Python 的数字人项目，它从 [HeyGem.ai](https://github.com/GuijiAI/HeyGem.ai) 中提取出来，它能够直接在 Linux 系统上运行，摆脱了对 Docker 和 Windows 系统的依赖。我们的目标是提供一个更易于部署和使用的数字人解决方案。

**如果你觉得这个项目对你有帮助，欢迎给我们 Star！**  
**如果运行过程中遇到问题，在查阅已有 Issue 后，在查阅 Google/baidu/ai 后，欢迎提交 Issues！**

## 主要特性

* 无需 Docker: 直接在 Linux 系统上运行，简化部署流程。
* 无需 Windows: 完全基于 Linux 开发和测试。
* Python 驱动: 使用 Python 语言开发，易于理解和扩展。
* 开发者友好: 易于使用和扩展。
* 完全离线。  

## 开始使用

### 环境
本项目包括 tts 和 face2face 两部分
* tts 部分支持 3.8，事实上有更高版本更好；
* face2face 部分支持且仅支持 3.8。


### 使用
把项目克隆到本地
```bash
# f2f
git clone https://github.com/Holasyb918/HeyGem-Linux-Python-Hack
cd HeyGem-Linux-Python-Hack
# 下载 f2f 模型
bash download.sh

# tts
git clone https://github.com/Holasyb918/tts-fish-speech
cd tts-fish-speech
# 下载 tts 模型
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5/
```

### 安装环境
请参考 [requirements.txt](https://github.com/Holasyb918/tts-fish-speech/blob/main/requirements.txt) 并结合你的实际环境来搭建环境，如果单个环境难以满足，tts 可以使用常规的环境，不要求 3.8，但你可能需要分步完成从 text 到数字人的整个流程。

#### 开始使用  
* repo 中已提供可以用于 demo 的音视频样例，代码可以直接运行。  
把你需要生成的文本放在 [example/text.txt](example/text.txt) 中，把要克隆的音色放在 [example/audio.wav](example/audio.wav) 中，然后运行以下命令：
#### command:  
```bash
bash inference_from_text.sh example/audio.wav example/text.txt example/video.mp4
#             音色 wav           TTS 文本            视频
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

## Contributing  
欢迎贡献！

## License
参考 heyGem.ai 的协议.
