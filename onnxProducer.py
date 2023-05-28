import tensorflow as tf
import numpy as np
import tf2onnx
import onnxruntime as ort
import onnx

tfk = tf.keras
tfkl = tf.keras.layers

# Define the neural network
#model = tf.keras.Sequential()
#model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)))
#model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#input_layer = tfkl.Input(shape=(8,), name='Input')
#hidden_layer1 = tfkl.Dense(units=12, activation='relu')(input_layer)
#hidden_layer2 = tfkl.Dense(units=4, activation='relu')(hidden_layer1)
#output_layer = tfkl.Dense(units=1, activation='linear')(hidden_layer2)
#model = tfk.Model(inputs=input_layer, outputs=output_layer, name='FFNN')

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=5, kernel_size=3, input_shape=(4, 10, 32)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=30, kernel_size=2, ), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
    #tf.keras.layers.GlobalMaxPool2D(),
    #tf.keras.layers.GlobalAveragePooling2D(),
    #tf.keras.layers.Flatten(name='layer_6'),
    tf.keras.layers.Dense(4, activation='softmax'),
])

model.summary()

# Define the input signature for the conversion
input_signature = [tf.TensorSpec([None, 4, 10, 32], tf.float32, name="input")]

# Convert the model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

# Save the ONNX model to a file
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Create a test input
test_input = np.ones((4, 10, 32), np.float32)

# Create an ONNX runtime session
sess = ort.InferenceSession("model.onnx")

#print(onnx_model)
#print('*****************************************************************')
#print('Model :\n\n{}'.format(onnx.helper.printable_graph(onnx_model.graph)))

# Run the inference using the ONNX model
#res = sess.run(None, {'input': test_input})
#print(res)