# This file runs `train.run()` from YOLOv5
# It assumes YOLO's home location is in the configuration files

import os
import sys
from configparser import ConfigParser

try:
    config = ConfigParser()
    cfg = os.path.join(os.path.dirname(__file__), 'cfg/yolo_home.cfg')
    config.read(cfg)
    sys.path.append(config.get('yolov5', 'home'))
    from yolov5 import train
except:
    print("Could not find YOLOv5! Make sure you provide its home path in the configuration file.")
    sys.exit()


if __name__ == "__main__":
    # Get train.cfg configurations
    cfg = os.path.join(os.path.dirname(__file__), 'cfg/train.cfg')
    config.read(cfg)

    # load
    data = config.get('load', 'data')
    weights = config.get('load', 'weights')

    # options
    imgsz = config.getint('options', 'imgsz')
    batch_size = config.getint('options', 'batch_size')
    epochs = config.getint('options', 'epochs')
    exist_ok = config.getboolean('options', 'exist_ok')

    # run-location
    project = config.get('run-location', 'project')
    name = config.get('run-location', 'name')

    # Perform trainig
    train.run(
        weights=weights,
        data=data,
        imgsz=imgsz,
        batch_size=batch_size,
        epochs=epochs,
        project=project,
        name=name,
        exist_ok=exist_ok
    )
