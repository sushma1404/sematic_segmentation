import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from matplotlib import pyplot as plt
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

EPOCHS = 32
BATCH_SIZE = 8
KEEP_PROB = 0.5
LEARNING_RATE = 0.001

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return w1, keep, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    with tf.variable_scope('VGGNet'):
        conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1, 1), padding='same',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(.001),
                                    name="conv_1x1")
        # tf.Print(conv_1x1, [tf.shape(conv_1x1)[:]])

        upsample1 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(2, 2), padding='same',
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(.001),
                                               name="upsample1")
        # tf.Print(upsample1, [tf.shape(upsample1)[:]])

        vgg_layer4_out = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1, 1), padding='same',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(.001),
                                          name="vgg_layer3_out")
        skip1 = tf.add(upsample1, vgg_layer4_out, name="skip1")
        # tf.Print(skip1, [tf.shape(skip1)[:]])

        upsample2 = tf.layers.conv2d_transpose(skip1, num_classes, 4, strides=(2, 2), padding='same',
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(.001),
                                               name="upsample2")
        # tf.Print(upsample2, [tf.shape(upsample2)[:]])

        vgg_layer3_out = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1, 1), padding='same',
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(.001),
                                          name="vgg_layer_3out")
        skip2 = tf.add(upsample2, vgg_layer3_out, name="skip2")
        # tf.Print(skip2, [tf.shape(skip2)[:]])

        final = tf.layers.conv2d_transpose(skip2, num_classes, 16, strides=(8, 8), padding='same',
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(.001),
                                           name="final")
        tf.Print(final, [tf.shape(final)[:]])

        return final
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="adam_logit")
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                        name="cross_loss")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'VGGNet')
    cross_entropy_loss = cross_entropy_loss + tf.reduce_sum(reg_ws)
    training_operation = optimizer.minimize(cross_entropy_loss, name="train_op")

    return logits, training_operation, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    sess.run(tf.global_variables_initializer())

    train_losses = []
    steps = 0

    for i in range(epochs):
        for images, gt_images in get_batches_fn(batch_size):
            steps += 1
            # feed = {input_image: images, correct_label: gt_images, keep_prob: KEEP_PROB, learning_rate: LEARNING_RATE}
            # _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed)
            #
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: images, correct_labels: gt_images,
                                                                         keep_prob: KEEP_PROB,
                                                                         learning_rate: LEARNING_RATE})

            if steps % 50 == 0:
                print('Epoch: {};Step: {};Loss: {}'.format(i, steps, loss))
        train_losses.append(loss)
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # correct_label = tf.placeholder(tf.float32)
        # learning_rate = tf.placeholder(tf.float32)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes), name="correct_label")
        learning_rate = tf.placeholder(dtype=tf.float32,  name="learning_rate")
        keep_prob = tf.placeholder(tf.float32)

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, optimizer, cost = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        restore = True # restore ckpt model
        train = True # can restore or start from beginning
        load_pb = False # To load frozen quantized graphs, only for inference

        saver = tf.train.Saver()

        if train:
            if restore:
                saver.restore(sess, tf.train.latest_checkpoint('./'))

                # train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_layer,
                #          correct_label, keep_prob, learning_rate)
                train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, optimizer, cost, image_input, correct_label, keep_prob,
                         learning_rate)
            else:

                sess.run(tf.global_variables_initializer())
                # train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_layer,
                #          correct_label, keep_prob, learning_rate)
                train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, optimizer, cost, image_input, correct_label, keep_prob,
                         learning_rate)

                # saver.save(sess, 'checkpoints/model8.ckpt')
            save_path = saver.save(sess, "model.ckpt")
            print("Model saved in file: %s" % save_path)
            # saver.export_meta_graph("/home/ubuntu/seg1/checkpoints/model.meta")
            tf.train.write_graph(sess.graph_def, '', 'model_text.pb', True)
            tf.train.write_graph(sess.graph_def, '', 'model_.pb', False)
            # TODO: Save inference data using helper.save_inference_samples

            # helper.save_inference_samples2(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_layer)
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        else:  # inference only

            if load_pb:
                sess, ops = gu.load_graph('eightbit.pb')
                g = sess.graph
                print("operations ", len(ops))
                print("graph names: \n", [n.name for n in g.as_graph_def().node])
                image_input = g.get_tensor_by_name('image_input:0')
                keep_prob = g.get_tensor_by_name('keep_prob:0')
                logits = g.get_tensor_by_name('adam_logit:0')
                helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

                # helper.save_inference_samples2(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_layer)
            else:
                saver.restore(sess, tf.train.latest_checkpoint('./'))

                # helper.save_inference_samples2(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_layer)
                helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # # TODO: Train NN using the train_nn function
        # train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, optimizer, cost, image_input, correct_label, keep_prob,
        #          learning_rate)
        #
        # # TODO: Save inference data using helper.save_inference_samples
        # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video



if __name__ == '__main__':
    run()
