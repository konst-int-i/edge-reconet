# Compilation for Edge-TPU

This submodule contains all the code necessary to get inference running on the
edge-tpu.

## Install
The following snippet will download the trained model and install it on the TPU.
Run the code below on the Edge TPU:
```
git clone https://github.com/konst-int-i/edge-tpu-video-style-transfer
cd edge_tpu
chmod +x install.sh
./install.sh
```

Then to run inference simply
```
python3 style_transfer.py
```

## TODO:
- [ ] Compile our model to work, inference-wise on the Edge TPU
- [ ] Compile the weights to work on the Edge TPU and put them on the internet
- [ ] Write install script that runs all necessary commands to install the model
- [ ] Write the Python program that runs inference on the TPU