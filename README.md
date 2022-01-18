# Edge TPU Video Style Transfer
Project to run video style transfer on Edge TPU


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

## Compilation steps I've tried
- [X] Does it work when you take away many of the resblocks
- [X] Does it work when you take away upsampling
- [ ] Does it work when the size is drastically reduced?
- [X] Does it work when you take away InstNorm? - Yes, it hates instnorm
- [X] Does it work on like 2 conv blocks? - Yes it does

So far it seems like it's the InstNorm the compiler doesn't like, but lets be sure

