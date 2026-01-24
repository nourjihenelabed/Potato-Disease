from cffi.cffi_opcode import CLASS_NAME
from fastapi import FastAPI , File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL_LAYER = tf.keras.layers.TFSMLayer("../models/2", call_endpoint='serving_default')
CLASS_NAMES=["Early Blight","Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Hello I am alive"
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read image
    image = read_file_as_image(await file.read())

    # image_pil = Image.fromarray(image).resize((256, 256))
    # image_resized = np.array(image_pil)

    # 3. Expand dims and convert to float32 (Required for Keras 3 inference)
    img_batch = np.expand_dims(image, 0).astype(np.float32)

    # 4. Run Prediction
    try:
        predictions_output = MODEL_LAYER(img_batch)

        # In 2026, TFSMLayer returns a dict.
        # We grab the first value in the dict regardless of its name ('output_0', 'dense_2', etc.)
        if isinstance(predictions_output, dict):
            # This line dynamically finds the correct output key
            first_key = list(predictions_output.keys())[0]
            predictions = predictions_output[first_key].numpy()
        else:
            predictions = predictions_output.numpy()

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            "class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        # This will print the actual error to your terminal if it fails again
        print(f"Prediction Error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
