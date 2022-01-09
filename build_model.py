import os
from models.reconet import build_reconet
from models.utils import save_model
from postprocessing.quantisation import quantise_model 

import tensorflow as tf

from timeit import default_timer as timer


if __name__=="__main__":
    model = build_reconet()
    model.compile(loss='mse', optimizer='Adam')
    inp = tf.random.uniform((1, 32, 32, 3), minval=0, maxval=1)
    print("Running model")
    before = timer()
    model.fit(inp, inp, epochs=1)
    res = model.predict(inp, batch_size=1)
    print("Finished running model")
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    model.save('saved_models/test_model')
    end = timer()
    print(f"Finished inference, {res.shape=}, time={end-before}, per sample={(end-before)/1}")

    print("quantising")
    quantised = quantise_model(model, 'test_model')

    save_model(quantised)
