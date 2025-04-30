import argparse
import gc
import json
import os

os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
import subprocess
import threading
import time
import traceback
import uuid
from enum import Enum
import queue
import shutil
from functools import partial

import cv2
import gradio as gr
from flask import Flask, request

import service.trans_dh_service
from h_utils.custom import CustomError
from y_utils.config import GlobalConfig
from y_utils.logger import logger


def write_video_gradio(
    output_imgs_queue,
    temp_dir,
    result_dir,
    work_id,
    audio_path,
    result_queue,
    width,
    height,
    fps,
    watermark_switch=0,
    digital_auth=0,
    temp_queue=None,
):
    output_mp4 = os.path.join(temp_dir, "{}-t.mp4".format(work_id))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    result_path = os.path.join(result_dir, "{}-r.mp4".format(work_id))
    video_write = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
    print("Custom VideoWriter init done")
    try:
        while True:
            state, reason, value_ = output_imgs_queue.get()
            if type(state) == bool and state == True:
                logger.info(
                    "Custom VideoWriter [{}]视频帧队列处理已结束".format(work_id)
                )
                logger.info(
                    "Custom VideoWriter Silence Video saved in {}".format(
                        os.path.realpath(output_mp4)
                    )
                )
                video_write.release()
                break
            else:
                if type(state) == bool and state == False:
                    logger.error(
                        "Custom VideoWriter [{}]任务视频帧队列 -> 异常原因:[{}]".format(
                            work_id, reason
                        )
                    )
                    raise CustomError(reason)
                for result_img in value_:
                    video_write.write(result_img)
        if video_write is not None:
            video_write.release()
        if watermark_switch == 1 and digital_auth == 1:
            logger.info(
                "Custom VideoWriter [{}]任务需要水印和数字人标识".format(work_id)
            )
            if width > height:
                command = 'ffmpeg -y -i {} -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(
                    audio_path,
                    output_mp4,
                    GlobalConfig.instance().watermark_path,
                    GlobalConfig.instance().digital_auth_path,
                    result_path,
                )
                logger.info("command:{}".format(command))
            else:
                command = 'ffmpeg -y -i {} -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(
                    audio_path,
                    output_mp4,
                    GlobalConfig.instance().watermark_path,
                    GlobalConfig.instance().digital_auth_path,
                    result_path,
                )
                logger.info("command:{}".format(command))
        elif watermark_switch == 1 and digital_auth == 0:
            logger.info("Custom VideoWriter [{}]任务需要水印".format(work_id))
            command = 'ffmpeg -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10" -c:a aac -crf 15 -strict -2 {}'.format(
                audio_path,
                output_mp4,
                GlobalConfig.instance().watermark_path,
                result_path,
            )
            logger.info("command:{}".format(command))
        elif watermark_switch == 0 and digital_auth == 1:
            logger.info("Custom VideoWriter [{}]任务需要数字人标识".format(work_id))
            if width > height:
                command = 'ffmpeg -loglevel warning -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(
                    audio_path,
                    output_mp4,
                    GlobalConfig.instance().digital_auth_path,
                    result_path,
                )
                logger.info("command:{}".format(command))
            else:
                command = 'ffmpeg -loglevel warning -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(
                    audio_path,
                    output_mp4,
                    GlobalConfig.instance().digital_auth_path,
                    result_path,
                )
                logger.info("command:{}".format(command))
        else:
            command = "ffmpeg -loglevel warning -y -i {} -i {} -c:a aac -c:v libx264 -crf 15 -strict -2 {}".format(
                audio_path, output_mp4, result_path
            )
            logger.info("Custom command:{}".format(command))
        subprocess.call(command, shell=True)
        print("###### Custom Video Writer write over")
        print(f"###### Video result saved in {os.path.realpath(result_path)}")
        result_queue.put([True, result_path])
        # temp_queue.put([True, result_path])
    except Exception as e:
        logger.error(
            "Custom VideoWriter [{}]视频帧队列处理异常结束，异常原因:[{}]".format(
                work_id, e.__str__()
            )
        )
        result_queue.put(
            [
                False,
                "[{}]视频帧队列处理异常结束，异常原因:[{}]".format(
                    work_id, e.__str__()
                ),
            ]
        )
    logger.info("Custom VideoWriter 后处理进程结束")


service.trans_dh_service.write_video = write_video_gradio


class VideoProcessor:
    def __init__(self):
        self.task = service.trans_dh_service.TransDhTask()
        self.basedir = GlobalConfig.instance().result_dir
        self.is_initialized = False
        self._initialize_service()
        print("VideoProcessor init done")

    def _initialize_service(self):
        logger.info("开始初始化 trans_dh_service...")
        try:
            time.sleep(5)
            logger.info("trans_dh_service 初始化完成。")
            self.is_initialized = True
        except Exception as e:
            logger.error(f"初始化 trans_dh_service 失败: {e}")

    def process_video(
        self, audio_file, video_file, watermark=False, digital_auth=False
    ):
        while not self.is_initialized:
            logger.info("服务尚未完成初始化，等待 1 秒...")
            time.sleep(1)
        work_id = str(uuid.uuid1())
        code = work_id
        temp_dir = os.path.join(GlobalConfig.instance().temp_dir, work_id)
        result_dir = GlobalConfig.instance().result_dir
        video_writer_thread = None
        final_result = None

        try:
            cap = cv2.VideoCapture(video_file)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            audio_path = audio_file
            video_path = video_file

            self.task.task_dic[code] = ""
            self.task.work(audio_path, video_path, code, 0, 0, 0, 0)

            result_path = self.task.task_dic[code][2]
            final_result_dir = os.path.join("result", code)
            os.makedirs(final_result_dir, exist_ok=True)
            os.system(f"mv {result_path} {final_result_dir}")
            os.system(
                f"rm -rf {os.path.join(os.path.dirname(result_path), code + '*.*')}"
            )
            result_path = os.path.realpath(
                os.path.join(final_result_dir, os.path.basename(result_path))
            )
            return result_path

        except Exception as e:
            logger.error(f"处理视频时发生错误: {e}")
            raise gr.Error(str(e))


if __name__ == "__main__":
    processor = VideoProcessor()

    inputs = [
        gr.File(label="上传音频文件/upload audio file"),
        gr.File(label="上传视频文件/upload video file"),
    ]
    outputs = gr.Video(label="生成的视频/Generated video")

    title = "数字人视频生成/Digital Human Video Generation"
    description = "上传音频和视频文件，即可生成数字人视频。/Upload audio and video files to generate digital human videos."

    demo = gr.Interface(
        fn=processor.process_video,
        inputs=inputs,
        outputs=outputs,
        title=title,
        description=description,
    )
    demo.queue().launch()
