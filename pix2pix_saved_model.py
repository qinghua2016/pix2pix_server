
import os.path
from pix2pix import *

tf.app.flags.DEFINE_string('checkpoint_dir', 'data/pix2pix_checkpoint',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', 'data/pix2pix_model',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
tf.app.flags.DEFINE_integer('image_size', 512,
                            """Needs to provide same value as in training.""")
FLAGS = tf.app.flags.FLAGS

NUM_CLASSES = 1000
NUM_TOP_CLASSES = 5


def export():

  with tf.Graph().as_default():
    image_size=512
    images = tf.placeholder(tf.float32, [None, image_size, image_size,3])
    model=pix2pix()
    # Run inference.
    outputs = model.sampler(images)
    saver = tf.train.Saver()

    with tf.Session() as sess:

      saver.restore(sess, 'data/pix2pix_checkpoint/demo_fm')
      # Export inference model.
      output_path = os.path.join(
          tf.compat.as_bytes(FLAGS.output_dir),
          tf.compat.as_bytes(str(FLAGS.model_version)))
      print('Exporting trained model to', output_path)
      builder = tf.saved_model.builder.SavedModelBuilder(output_path)
      inputs_tensor_info = tf.saved_model.utils.build_tensor_info(images)
      outputs_tensor_info = tf.saved_model.utils.build_tensor_info(
          outputs)


      prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'images': inputs_tensor_info},
              outputs={
                  'outputs': outputs_tensor_info,

              },
              method_name=tf.saved_model.signature_constants.REGRESS_METHOD_NAME
          ))


      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               prediction_signature}
       )

      builder.save()
      print('Successfully exported model to %s' % FLAGS.output_dir)


export()