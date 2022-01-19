import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from io_flow import readFlow
from PIL import Image
from skimage import transform
import os


class MPIDataSet:
    def __init__(self, path, args):
        """
        looking at the "clean" subfolder for images, might change to "final" later
        root_dir -> path to the location where the "training" folder is kept inside the MPI folder
        """
        self.path = path
        self.args = args
        self.transform = transform
        self.dirlist = os.listdir(self.path.joinpath("clean"))
        self.dirlist.sort()
        self.dirlist = [item for item in self.dirlist if item.find("bandage_1") < 0]
        self.numlist = []
        for folder in self.dirlist:
            self.numlist.append(
                len(os.listdir(self.path.joinpath("clean").joinpath(folder)))
            )
        self.idx = 0

    def __len__(self):
        return sum(self.numlist) - len(self.numlist)

    def __iter__(self):
        return self

    # def __getitem__(self, idx):
    def __next__(self):

        """
        idx must be between 0 to len-1
        assuming flow[0] contains flow in x direction and flow[1] contains flow in y
        """
        self.idx += 1
        float_conv_factor = 255

        for i in range(0, len(self.numlist)):
            folder = self.dirlist[i]
            path = self.path.joinpath("clean").joinpath(folder)
            occpath = self.path.joinpath("occlusions").joinpath(folder)
            flowpath = self.path.joinpath("flow").joinpath(folder)
            if self.idx < (self.numlist[i] - 1):
                num1 = toString(self.idx + 1)
                num2 = toString(self.idx + 2)

                img1 = Image.open(path.joinpath(f"frame_{num1}.png")).resize(
                    (self.args.width, self.args.height), Image.BILINEAR
                )
                img2 = Image.open(path.joinpath(f"frame_{num2}.png")).resize(
                    (self.args.width, self.args.height), Image.BILINEAR
                )
                mask = Image.open(occpath.joinpath(f"frame_{num1}.png")).resize(
                    (self.args.width, self.args.height), Image.BILINEAR
                )  # note: changed from bilinear interpolation

                flow = readFlow(str(flowpath.joinpath(f"frame_{num1}.flo")))

                img1 = (
                    tf.convert_to_tensor(np.array(img1), dtype=tf.float32)
                    / float_conv_factor
                )
                img1 = tf.transpose(img1, (1, 0, 2))
                img2 = (
                    tf.convert_to_tensor(np.array(img2), dtype=tf.float32)
                    / float_conv_factor
                )
                img2 = tf.transpose(img2, (1, 0, 2))
                h, w, c = flow.shape

                flow = tf.convert_to_tensor(np.array(flow), dtype=tf.float32)
                flow = tf.transpose(flow, (1, 0, 2))
                flow = tf.image.resize(
                    images=flow, size=(self.args.width, self.args.height)
                )
                multiplier = tf.constant(
                    [flow.shape[0] / w, flow.shape[1] / h], dtype=tf.float32
                )
                flow *= multiplier

                ##take no occluded regions to compute
                # need to convert to numpy array first

                mask = np.asarray(mask) / 255
                mask = 1 - mask
                mask = np.where((mask < 0.99), 0.0, 1.0)
                # now convert to tensor
                mask = tf.convert_to_tensor(np.array(mask), dtype=tf.float32)
                mask = tf.expand_dims(mask, axis=0)
                mask = tf.transpose(mask, (2, 1, 0))
                break
            self.idx -= self.numlist[i] - 1
            # IMG2 should be at t in IMG1 is at T-1
        return (img1, img2, flow, mask)

    def __call__(self, *args, **kwargs):
        return self


def toString(num):
    string = str(num)
    while len(string) < 4:
        string = "0" + string
    return string


def read_style_image(args):
    style_img = Image.open(args.style_name)
    style_img = style_img.resize((args.width, args.height), Image.BILINEAR)
    style_img = tf.convert_to_tensor(np.array(style_img))
    style_img = tf.transpose(style_img, (1, 0, 2))
    style_img = tf.cast(tf.expand_dims(style_img / 255, axis=0), dtype=tf.float32)
    return style_img
