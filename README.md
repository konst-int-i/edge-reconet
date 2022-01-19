# Edge TPU Video Style Transfer
Project to run video style transfer on Edge TP


## Installation

This has been tested in `python=3.9.X`. Please create a virtual environment and install the requirements: 

Windows
```
py -m venv edge
.\edge\Scripts\activate
pip install -r requirements.txt
```

Mac/Ubuntu
```
python3 -m venv edge
source edge/bin/activate
pip install -r requirements.txt
```

## Run instructions

#### Configurations 

All critical run defaults for training and inference can be found in our argument parser in
`reconet_tensorflow/utils/parser.py`. 

#### Training

Full training can be run using: 

```
python3 reconet_tensorflow/train.py
```

During development, you can run the debug mode by passing the flag 

```
python3 reconet_tensorflow/train.py --debug True
```


#### Webcam version

You can run the style inference via your webcam on your local machine (note that you might need to give permission 
on your machine) using the default arguments. Note that the default aspect ratios are set to that of MacBook 
webcams

```
python3 reconet_tensorflow/video_demo.py
```

#### Style video

```
python3 reconet_tensorflow/video_demo.py --video-input sample_vid.mp4 --video-output styled_video.mp4
```

You can also specify the input/output resolution using (here 512x216 for input, 1080x760 for output)

```
python3 reconet_tensorflow/video_demo.py --input-resolution 512, 360 --output-resolution 1080, 760
```


![Alt Text](kings_parade.gif)