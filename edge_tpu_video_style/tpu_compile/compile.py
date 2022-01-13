from quantisation import quantise_model
from utils import save_model

def compile_model(model):
    model = quantise_model(model)
    save_model(model)


if __name__ == "__main__":
    compile_model('reconet_model')