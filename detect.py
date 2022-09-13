# This file `runs detect.run()` from YOLOv5
# It assumes YOLO's home location is in the configuration files

import os
import sys
from configparser import ConfigParser

try:
    config = ConfigParser()
    cfg = os.path.join(os.path.dirname(__file__), 'cfg/detect.cfg')
    config.read(cfg)
    sys.path.append(config.get('yolov5', 'home'))
    from yolov5 import detect
except:
    print("Could not find YOLOv5! Make sure you provide its home path in the configuration file.")
    sys.exit()


if __name__ == "__main__":
    # Get detect.cfg configurations
    data = config.get('load', 'data')
    weights = config.get('load', 'weights')
    source = config.get('load', 'source')

    imgsz = config.getint('options', 'imgsz')
    conf_thres = config.getfloat('options', 'conf_thres')
    iou_thres = config.getfloat('options', 'iou_thres')
    save_txt = config.getboolean('options', 'save_txt')
    save_conf = config.getboolean('options', 'save_conf')
    exist_ok = config.getboolean('options', 'exist_ok')

    project = config.get('run-location', 'project')
    name = config.get('run-location', 'name')

    # Perform detection
    detect.run(
        weights=weights,
        source=source,
        data=data,
        imgsz=(imgsz, imgsz),
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        save_txt=save_txt,
        save_conf=save_conf,
        project=project,
        name=name,
        exist_ok=exist_ok
    )
