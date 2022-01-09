import argparse

parser = argparse.ArgumentParser(description="Style Transfer")
parser.add_argument(
    "--batch-size",
    type=int,
    default=4,
    metavar="N",
    help="input batch size for training (default: 1)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1,
    metavar="N",
    help="input  batch size for test (default: 1)",
)
parser.add_argument(
    "--phase", type=str, default="train", help="train, test, predict(default:train)"
)
parser.add_argument("--epochs", type=int, default=100, help="epoch (default:100)")
parser.add_argument("--path", type=str, default="MPI-Sintel-complete", help="path")
parser.add_argument(
    "--log-interval", type=int, default=1, help="log interval(default:1)"
)
parser.add_argument(
    "--lr", type=float, default=0.001, help="learning rate(default:0.0001)"
)
parser.add_argument(
    "--save-model", action="store_true", default=True, help="save the current Model"
)
parser.add_argument(
    "--save-directory",
    type=str,
    default="trained_models",
    help="learnt models are saving here",
)
parser.add_argument(
    "--LAMBDA-O", type=float, default=1e1, help="output_temp_loss hyperparameter"
)
parser.add_argument(
    "--LAMBDA-F", type=float, default=1e-1, help="feature_temp_loss hyperparameter"
)
parser.add_argument(
    "--ALPHA", type=float, default=1e0, help="content_loss hyperparameter"
)
parser.add_argument("--BETA", type=float, default=1e5, help="style_loss hyperparameter")
parser.add_argument("--GAMMA", type=float, default=1e-6, help="reg_loss hyperparameter")
parser.add_argument("--model-name", type=str, default="", help="model name")
parser.add_argument(
    "--style-name",
    type=str,
    default="data/style_images/flower.jpeg",
    help="style image name",
)
parser.add_argument(
    "--output-model", type=bool, default=False, help="True: generate output model name"
)
parser.add_argument("--is-schedular", type=bool, default=True, help="")
parser.add_argument("--shuffle_buffer", type=int, default=1024)
parser.add_argument("--width", default=512)
parser.add_argument("--height", default=216)
