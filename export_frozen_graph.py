import tensorflow as tf

tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')

import model

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    import os


    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        _, _ = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            # frozen graph
            output_graph = "east_det.pb"
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    tf.app.run()
