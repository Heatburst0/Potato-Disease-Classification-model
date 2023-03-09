from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

Bucket_name="potato-disease-tf-model-2"
class_names=["Early Blight","Late Blight","Healthy"]
model=None
def download_blob(bucket_name,source_blob_name,dest_file):
    storage_client=storage.Client()
    bucket=storage_client.get_bucket(bucket_name)
    blob=bucket.blob(source_blob_name)
    blob.download_to_filename(dest_file)

def predict(request):
    global model
    if model is None:
        download_blob(
            Bucket_name,
            "models/potatoes.h5",
            "/tmp/potatoes.h5"
        )
        model=tf.keras.models.load_model("/tmp/potatoes.h5")
    image=request.files["file"]
    image=np.array(Image.open(image).convert("RGB").resize((256,256)))
    image=image/255
    img_array=tf.expand_dims(image,0)
    pred=model.predict(img_array)
    res = class_names[np.argmax(pred[0])]
    confidence = np.max(pred[0])
    return {'class': res,
            'confidence': float(confidence)
            }

