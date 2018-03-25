#!/usr/bin/env python2.7

from __future__ import print_function
from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import numpy as np
import cv2
import scipy
import os
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', 'data/test_data/1.JPG', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS
def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  data = imread(FLAGS.image)
  data = data / 127.5 - 1.
  image_size=512
  sample=[]
  sample.append(data)
  sample_image = np.asarray(sample).astype(np.float32)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'pix2pix'
  request.model_spec.signature_name =tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  request.inputs['images'].CopyFrom(
      tf.contrib.util.make_tensor_proto(sample_image, shape=[1, image_size, image_size,3]))
  result_future = stub.Predict.future(request, 5.0)  # 5 seconds
  response = np.array(
    result_future.result().outputs['outputs'].float_val)
  out=(response.reshape((512,512,3))+1)*127.5
  out= cv2.cvtColor(out.astype(np.float32), cv2.COLOR_BGR2RGB)
  cv2.imwrite('data/test_result/' + '1.jpg', out)




if __name__ == '__main__':
  tf.app.run()
