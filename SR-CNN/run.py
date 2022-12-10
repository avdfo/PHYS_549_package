import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

import source.neuralnet as nn
import source.datamanager as dman
import source.tf_process as tfp


def main():

    srnet = nn.SRNET()

    dataset = dman.DataSet()

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()

    # Execute training the neural network
    tfp.training(sess=sess, neuralnet=srnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, batch_size_val=FLAGS.batch_val)
    # Execute testing the neural network on the test dataset
    tfp.testing(sess=sess, neuralnet=srnet, saver=saver, dataset=dataset)
    # Execute applying the trained neural network on actual ARPES data in test_exp folder
    tfp.testing_exp(sess=sess, neuralnet=srnet, saver=saver, dataset=dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Specify epoch number for training
    parser.add_argument('--epoch', type=int, default=200, help='-')
    # Specify training batch size
    parser.add_argument('--batch', type=int, default=16, help='-')
    # Specify validation batch size.
    # Note that the ratio between training and validation batch size should match the ratio between training
    # and validation dataset size.
    parser.add_argument('--batch_val', type=int, default=4, help='-')

    FLAGS, unparsed = parser.parse_known_args()

    main()
