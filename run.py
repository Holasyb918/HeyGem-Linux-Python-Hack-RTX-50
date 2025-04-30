import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from enum import Enum

import queue
import cv2
from flask import Flask, request

import service.trans_dh_service

from h_utils.custom import CustomError
from y_utils.config import GlobalConfig
from y_utils.logger import logger


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=(argparse.ArgumentDefaultsHelpFormatter)
    )

    parser.add_argument(
        "--audio_path",
        type=str,
        default="example/audio.wav",
        help="path to local audio file",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="example/video.mp4",
        help="path to local video file",
    )
    opt = parser.parse_args()
    return opt


def write_video(
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
        exit(0)
        result_queue.put([True, result_path])
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


service.trans_dh_service.write_video = write_video


def main():
    opt = get_args()
    if not os.path.exists(opt.audio_path):
        audio_url = "example/audio.wav"
    else:
        audio_url = opt.audio_path

    if not os.path.exists(opt.video_path):
        video_url = "example/video.mp4"
    else:
        video_url = opt.video_path
    sys.argv = [sys.argv[0]]
    task = service.trans_dh_service.TransDhTask()
    time.sleep(10) # somehow, this works...

    code = "1004"
    task.work(audio_url, video_url, code, 0, 0, 0, 0)


if __name__ == "__main__":
    main()

# python run.py
# python run.py --audio_path example/audio.wav --video_path example/video.mp4
