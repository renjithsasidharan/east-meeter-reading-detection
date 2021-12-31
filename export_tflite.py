import tensorflow as tf

IMG_SIZE = 320

assert IMG_SIZE % 32 == 0, 'size has to be multiple of 32'

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file='east_det.pb', 
    input_arrays=['input_images'],
    output_arrays=['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'],
    input_shapes={'input_images': [1, IMG_SIZE, IMG_SIZE, 3]}
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
open('east_det_fp16.tflite', 'wb').write(tflite_model)


converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file='east_det.pb', 
    input_arrays=['input_images'],
    output_arrays=['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'],
    input_shapes={'input_images': [1, IMG_SIZE, IMG_SIZE, 3]}
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
open('east_det_dr.tflite', 'wb').write(tflite_model)