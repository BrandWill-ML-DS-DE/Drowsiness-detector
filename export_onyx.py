# export_onnx.py
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("eye_cnn.h5")

spec = (tf.TensorSpec((None,64,64,1), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

with open("eye_cnn.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

print("ONNX export complete.")

