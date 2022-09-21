# Train, Validate an Test Neural Nets for Autism Spectrum Disorder

This repository serves as an isolated place to train, validate and test NNs that are being used in the web app [repository](https://github.com/ASDDataMining/therapy-aid-tool). For now, it uses [YOLOv5]() to detect interactions between actors in a ASD therapy session.



## Install

1. Clone YOLOv5 and annotate its path.
2. Clone this repo and install its requirements.
    1. Update YOLOv5 `path` variable in the config file.

```bash
# Clone YOLOv5 and store its home location since it is not pip installable
git clone https://github.com/ultralytics/yolov5.git

# Clone and install this repo
git clone https://github.com/ASDDataMining/therapy-aid-nn.git
cd therapy-aid-nn
# python3 -m venv venv; source venv/bin/activate  # optional: python virtual environment
pip install -r requirements.txt
# Then go to cfg/yolo_location.cfg and change yolov5's path
```

## Usage
* After installation you can adjust the configuration files to your needs and run any of the corresponding `.py` files: `train.py`, `val.py` and `detect.py`.
* If wandb was installed correctly you can make logs in it
* For downloading labelbox images with the script provided you must create a `.env` file at the repo top-level and edit `LABELBOX_API=<your-labelbox-api>`.