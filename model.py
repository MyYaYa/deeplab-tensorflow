import tensorflow as tf
import numpy as np
from utils import inverse_channel, decode_mask


class DeepLab(object):
    def __init__(self, images, labels, num_class, max_iter, weight_decay, power, base_lr, image_mean):
        self.images = images
        self.labels = labels
        self.num_class = num_class
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.power = power
        self.base_lr = base_lr
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.image_mean = image_mean


    def vgg_model(self, images, labels, is_training, num_class):
        with tf.device("/cpu:0"):
            with tf.name_scope("block1"):
                conv1_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=[3, 3],
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv1_1")

                conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=[3, 3],
                                           padding='same', activation=tf.nn.relu, use_bias=True,
                                           name="conv1_2")

                pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[3, 3], strides=[2, 2],
                                                padding="same", name="pool1")

            with tf.name_scope("block2"):
                conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3],
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv2_1")

                conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=[3, 3],
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv2_2")

                pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[3, 3], strides=[2, 2],
                                                padding="same", name="pool2")

            with tf.name_scope("block3"):
                conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3],
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv3_1")

                conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=[3, 3],
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv3_2")

                conv3_3 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=[3, 3],
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv3_3")

                pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[3, 3], strides=[2, 2],
                                                padding="same", name="pool3")

            with tf.name_scope("block4"):
                conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3],
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv4_1")

                conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=[3, 3],
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv4_2")

                conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=[3, 3],
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv4_3")

                pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[3, 3], strides=[1, 1],
                                                padding="same", name="pool4")

            with tf.name_scope("block5"):
                conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3], dilation_rate=(2, 2),
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv5_1")

                conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[3, 3], dilation_rate=(2, 2),
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv5_2")

                conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=[3, 3], dilation_rate=(2, 2),
                                           padding="same", activation=tf.nn.relu, use_bias=True,
                                           name="conv5_3")

                pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[3, 3], strides=[1, 1],
                                                padding="same", name="pool5")

            # hole = 6
            with tf.name_scope("aspp1"):
                fc6_1 = tf.layers.conv2d(inputs=pool5, filters=1024, kernel_size=[3, 3], dilation_rate=(6, 6),
                                         padding="same", activation=tf.nn.relu, use_bias=True, name="fc6_1")

                drop6_1 = tf.layers.dropout(inputs=fc6_1, rate=0.5, training=is_training, name="drop6_1")

                fc7_1 = tf.layers.conv2d(inputs=drop6_1, filters=1024, kernel_size=[1, 1],
                                         padding="same", activation=tf.nn.relu, use_bias=True, name="fc7_1")

                drop7_1 = tf.layers.dropout(inputs=fc7_1, rate=0.5, training=is_training, name="drop7_1")

                fc8_1 = tf.layers.conv2d(inputs=drop7_1, filters=num_class, kernel_size=[1, 1],
                                         padding="same", use_bias=True, name="fc8_1")

            # hole = 12
            with tf.name_scope("aspp2"):
                fc6_2 = tf.layers.conv2d(inputs=pool5, filters=1024, kernel_size=[3, 3], dilation_rate=(12, 12),
                                         padding="same", activation=tf.nn.relu, use_bias=True, name="fc6_2")
                drop6_2 = tf.layers.dropout(inputs=fc6_2, rate=0.5, training=is_training, name="drop6_2")
                fc7_2 = tf.layers.conv2d(inputs=drop6_2, filters=1024, kernel_size=[1, 1],
                                         padding="same", activation=tf.nn.relu, use_bias=True, name="fc7_2")
                drop7_2 = tf.layers.dropout(inputs=fc7_2, rate=0.5, training=is_training, name="drop7_2")
                fc8_2 = tf.layers.conv2d(inputs=drop7_2, filters=num_class, kernel_size=[1, 1],
                                         padding="same", use_bias=True, name="fc8_2")

            # hole = 18
            with tf.name_scope("aspp3"):
                fc6_3 = tf.layers.conv2d(inputs=pool5, filters=1024, kernel_size=[3, 3], dilation_rate=(18, 18),
                                         padding="same", activation=tf.nn.relu, use_bias=True, name="fc6_3")
                drop6_3 = tf.layers.dropout(inputs=fc6_3, rate=0.5, training=is_training, name="drop6_3")
                fc7_3 = tf.layers.conv2d(inputs=drop6_3, filters=1024, kernel_size=[1, 1],
                                         padding="same", activation=tf.nn.relu, use_bias=True, name="fc7_3")
                drop7_3 = tf.layers.dropout(inputs=fc7_3, rate=0.5, training=is_training, name="drop7_3")
                fc8_3 = tf.layers.conv2d(inputs=drop7_3, filters=num_class, kernel_size=[1, 1],
                                         padding="same", use_bias=True, name="fc8_3")

            # hole = 24
            with tf.name_scope("aspp4"):
                fc6_4 = tf.layers.conv2d(inputs=pool5, filters=1024, kernel_size=[3, 3], dilation_rate=(24, 24),
                                         padding="same", activation=tf.nn.relu, use_bias=True, name="fc6_4")
                drop6_4 = tf.layers.dropout(inputs=fc6_4, rate=0.5, training=is_training, name="drop6_4")
                fc7_4 = tf.layers.conv2d(inputs=drop6_4, filters=1024, kernel_size=[1, 1],
                                         padding="same", activation=tf.nn.relu, use_bias=True, name="fc7_4")
                drop7_4 = tf.layers.dropout(inputs=fc7_4, rate=0.5, training=is_training, name="drop7_4")
                fc8_4 = tf.layers.conv2d(inputs=drop7_4, filters=num_class, kernel_size=[1, 1],
                                         padding="same", use_bias=True, name="fc8_4")

            with tf.name_scope("sum"):
                fc8 = tf.add_n(inputs=[fc8_1, fc8_2, fc8_3, fc8_4], name="fc8")


        return fc8

    def get_all_trainable(self):
        all_trainable = tf.trainable_variables()
        fc8_trainable = [v for v in all_trainable if 'fc8' in v.name]
        conv_trainable = [v for v in all_trainable if 'fc8' not in v.name]
        conv_w_trainable = [v for v in conv_trainable if "kernel" in v.name]  # lr * 1.0
        conv_b_trainable = [v for v in conv_trainable if "bias" in v.name]  # lr * 2.0
        fc8_w_trainable = [v for v in fc8_trainable if "kernel" in v.name]  # lr * 10.0
        fc8_b_trainable = [v for v in fc8_trainable if "bias" in v.name]  # lr * 20.0
        assert (len(all_trainable) == len(fc8_trainable) + len(conv_trainable))
        assert (len(conv_trainable) == len(conv_w_trainable) + len(conv_b_trainable))
        assert (len(fc8_trainable) == len(fc8_w_trainable) + len(fc8_b_trainable))
        return [all_trainable, conv_w_trainable, conv_b_trainable, fc8_w_trainable, fc8_b_trainable]

    def loss(self, all_trainable, model_output, labels):
        shrink_label = tf.image.resize_nearest_neighbor(labels, tf.stack(model_output.get_shape()[1:3]))
        flat_output = tf.reshape(model_output, [-1, self.num_class])
        flat_label = tf.reshape(shrink_label, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_output, labels=flat_label)

        l2_losses = [self.weight_decay * tf.nn.l2_loss(v) for v in all_trainable if 'kernel' in v.name]
        reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
        tf.summary.scalar("training_loss", reduced_loss, collections=["train_summary"])

        with tf.variable_scope("confusion_op"):
            prediction = tf.argmax(model_output, axis=3)
            prediction = tf.reshape(prediction, [-1])
            label = tf.squeeze(shrink_label, squeeze_dims=[3])
            label = tf.reshape(label, [-1])
            confusion_matrix = tf.get_variable(name="confusion_matrix", shape=(self.num_class, self.num_class), dtype=tf.int32)
            assign_op = tf.assign_add(confusion_matrix, tf.confusion_matrix(label, prediction, num_classes=self.num_class))

        return reduced_loss, assign_op

    def apply_gradient(self, reduced_loss, step, trainable_v):
        learning_rate = tf.scalar_mul(self.base_lr, tf.pow((1 - step / self.max_iter), self.power))
        opt_conv_w = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        opt_conv_b = tf.train.MomentumOptimizer(learning_rate * 2.0, momentum=0.9)
        opt_fc8_w = tf.train.MomentumOptimizer(learning_rate * 10.0, momentum=0.9)
        opt_fc8_b = tf.train.MomentumOptimizer(learning_rate * 20.0, momentum=0.9)

        all_trainable, conv_w_trainable, conv_b_trainable, fc8_w_trainable, fc8_b_trainable = trainable_v
        grads = tf.gradients(reduced_loss, conv_w_trainable + conv_b_trainable + fc8_w_trainable + fc8_b_trainable)
        len1 = len(conv_w_trainable)
        len2 = len(conv_b_trainable)
        len3 = len(fc8_w_trainable)
        len4 = len(fc8_b_trainable)
        print(len1, len2, len3, len4)
        grads_conv_w = grads[:len1]
        grads_conv_b = grads[len1:(len1 + len2)]
        grads_fc8_w = grads[(len1 + len2):(len1 + len2 + len3)]
        grads_fc8_b = grads[(len1 + len2 + len3):]
        train_op_conv_w = opt_conv_w.apply_gradients(zip(grads_conv_w, conv_w_trainable))
        train_op_conv_b = opt_conv_b.apply_gradients(zip(grads_conv_b, conv_b_trainable))
        train_op_fc8_w = opt_fc8_w.apply_gradients(zip(grads_fc8_w, fc8_w_trainable))
        train_op_fc8_b = opt_fc8_b.apply_gradients(zip(grads_fc8_b, fc8_b_trainable))
        train_op = tf.group(train_op_conv_w, train_op_conv_b, train_op_fc8_w, train_op_fc8_b)
        return train_op

    def validation_summary(self, c_m):
        with tf.name_scope("evaluation_op"):
            c_m = tf.cast(c_m, dtype=tf.float32)
            diag = tf.matrix_diag_part(c_m)
            diag_sum = tf.reduce_sum(diag)
            total_sum = tf.reduce_sum(c_m)
            # accuracy
            accuracy = tf.div(tf.cast(diag_sum, dtype=tf.float32), tf.cast(total_sum, dtype=tf.float32))
            sum_accuracy = tf.summary.scalar("accuracy", accuracy)
            # avg precision
            col_sum = tf.reduce_sum(c_m, axis=0)
            indices_ap = tf.squeeze(tf.where(tf.greater(col_sum, 0)), 1)
            col_sum_ap = tf.gather(col_sum, indices_ap)
            diag_ap = tf.gather(diag, indices_ap)
            avg_precision = tf.reduce_mean(tf.div(tf.cast(diag_ap, tf.float32), tf.cast(col_sum_ap, tf.float32)))
            sum_avg_precision = tf.summary.scalar("avg_precision", avg_precision)
            # avg recall
            row_sum = tf.reduce_sum(c_m, axis=1)
            indices_ar = tf.squeeze(tf.where(tf.greater(row_sum, 0)), 1)
            row_sum_ar = tf.gather(row_sum, indices_ar)
            diag_ar = tf.gather(diag, indices_ar)
            avg_recall = tf.reduce_mean(tf.div(tf.cast(diag_ar, tf.float32), tf.cast(row_sum_ar, tf.float32)))
            sum_avg_recall = tf.summary.scalar("avg_recall", avg_recall)
            # avg jaccard
            union = tf.subtract(tf.add(row_sum, col_sum), diag)
            indices_aj = tf.squeeze(tf.where(tf.greater(union, 0)), 1)
            union_aj = tf.gather(union, indices_aj)
            diag_aj = tf.gather(diag, indices_aj)
            avg_jaccard = tf.reduce_mean(tf.div(tf.cast(diag_aj, tf.float32), tf.cast(union_aj, tf.float32)))
            sum_avg_jaccard = tf.summary.scalar("avg_jaccard", avg_jaccard)

            evaluation_op = tf.summary.merge([sum_accuracy, sum_avg_precision, sum_avg_recall, sum_avg_jaccard])

            return evaluation_op

    def image_summary(self, model_output):
        with tf.name_scope("image_summary"):
            # process predictions: for visualization
            fc8_up = tf.image.resize_bilinear(model_output, size=self.images.get_shape()[1:3])
            fc8_up = tf.argmax(fc8_up, axis=3)
            pred = tf.expand_dims(fc8_up, axis=3)
            # image summary
            images_summary = tf.py_func(inverse_channel, [self.images, self.image_mean], tf.uint8)
            labels_summary = tf.py_func(decode_mask, [self.labels], tf.uint8)
            preds_summary = tf.py_func(decode_mask, [pred], tf.uint8)
            total_summary = tf.summary.image('images',
                                             tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                                             max_outputs=10)
            return total_summary


    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g,0)

                grads.append(expanded_g)

            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def train(self, train_record, iterator, val_record, iterator_val):
        # images = tf.placeholder(dtype=tf.float32, shape=(None, 321, 321, 3))
        # labels = tf.placeholder(dtype=tf.int32, shape=(None, 321, 321, 1))
        step = tf.placeholder(dtype=tf.float32)
        print(step)
        raw_output = self.vgg_model(self.images, self.labels, self.is_training, self.num_class)
        all_v = self.get_all_trainable()
        all_trainable, conv_w_trainable, conv_b_trainable, fc8_w_trainable, fc8_b_trainable = all_v
        print(conv_w_trainable)
        loss, assign_op = self.loss(all_trainable, raw_output, self.labels)
        train_op = self.apply_gradient(loss, step ,all_v)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

            data_dict = np.load("initial.npy", encoding="bytes").item()
            # load pre-trained model weight
            for op_name in data_dict:
                if "fc8" in op_name:
                    print("Not initialize %s" % op_name)
                    continue
                with tf.variable_scope(op_name, reuse=True):
                    for param_name, data in data_dict[op_name].items():
                        try:
                            if param_name == b'weights':
                                var = tf.get_variable("kernel")
                                sess.run(var.assign(data))
                            else:
                                var = tf.get_variable("bias")
                                sess.run(var.assign(data))
                        except ValueError:
                            raise

            # saver = tf.train.Saver(max_to_keep=10)

            # summary writer
            train_summary = tf.summary.merge_all("train_summary")
            print(train_summary)
            train_writer = tf.summary.FileWriter("./log/train_20171113", sess.graph)

            image_sum = self.image_summary(raw_output)
            c_m = tf.placeholder(dtype=tf.int32, shape=(self.num_class, self.num_class))
            print(c_m)
            evaluation_op = self.validation_summary(c_m)

            sess.run(init)
            sess.run(iterator.initializer, feed_dict={train_record: 'dataset/train.record'})
            step_t = 0
            print("training begin")
            while step_t < self.max_iter:
                im, lb = sess.run(iterator.get_next())
                feed_dict = {self.images: im, self.labels: lb, step: step_t, self.is_training: True}
                summary, _ = sess.run([train_summary, train_op], feed_dict=feed_dict)
                train_writer.add_summary(summary, step_t + 1)

                if (step_t + 1) % 1 == 0:
                    # print("Save model for iteration %s" % (step + 1))
                    # save_path = saver.save(sess, "./model/20171113/params-deeplab-%s" % (step + 1))

                    print("View performance on Validation dataset")
                    sess.run(iterator_val.initializer, feed_dict={val_record: 'dataset/val.record'})
                    scope = tf.get_variable_scope()
                    scope.reuse_variables()
                    init = tf.constant(0, shape=[self.num_class, self.num_class], dtype=tf.int32)
                    sess.run(tf.assign(tf.get_variable("confusion_op/confusion_matrix", dtype=tf.int32), init))
                    c_matrix = None
                    while True:
                        try:
                            im_val, lb_val = sess.run(iterator_val.get_next())
                            feed_dict_val = {self.images: im_val, self.labels: lb_val, self.is_training: False}
                            c_matrix, summary_img = sess.run([assign_op, image_sum], feed_dict=feed_dict_val)
                            train_writer.add_summary(summary_img, step_t + 1)
                        except tf.errors.OutOfRangeError:
                            print(c_matrix)
                            print("end of Validation dataset\n")
                            summary_eval = sess.run(evaluation_op, feed_dict={c_m: c_matrix})
                            train_writer.add_summary(summary_eval, step_t + 1)
                            break
                step_t = step_t + 1

            print("training end")