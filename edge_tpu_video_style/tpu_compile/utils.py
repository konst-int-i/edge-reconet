
def save_model(quantised, name="style_transfer.tflite"):
    with open(f"saved_models/{name}", "wb") as f:
        f.write(quantised)
