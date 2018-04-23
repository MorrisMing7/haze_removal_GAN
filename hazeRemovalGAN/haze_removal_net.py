# coding=utf-8



import os
import glob
import collections
import math
import time
from ops import *
import numpy as np
import scipy.misc as misc


EPS = 1e-12

Examples = collections.namedtuple("Examples",
                                  "paths,inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model",
        "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, "
        "gen_loss_GAN, content_loss_L1, gen_grads_and_vars, train")



class haze_removal_net:

    def __init__(self,im_size,batch_size,max_epoch,
                 gan_weight,l1_weight,adam_lr,adam_beta1,
                 train_dir,test_dir,log_dir,ckt_dir,display_dir,
                 freq_summary,freq_trace,freq_display,freq_process,freq_save):
        self.train_dir=train_dir
        self.test_dir = test_dir
        self.im_size=im_size
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.gan_weight = gan_weight
        self.l1_weight = l1_weight
        self.adam_lr = adam_lr
        self.adam_beta1 = adam_beta1
        self.log_dir = log_dir
        self.ckt_dir = ckt_dir
        self.display_dir = display_dir
        self.freq_summary =freq_summary
        self.freq_trace = freq_trace
        self.freq_display = freq_display
        self.freq_process = freq_process
        self.freq_save = freq_save
        self.have_trained =False

    def generator(self,generator_inputs, generator_outputs_channels):
        layers = []
        ngf = 64
        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = conv(generator_inputs, ngf, stride=2)
            layers.append(output)

        layer_specs = [
            ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(rectified, out_channels, stride=2)
                output = batchnorm(convolved)
                layers.append(output)

        layer_specs = [
            (ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    inputs = layers[-1]
                else:
                    inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(inputs)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = deconv(rectified, out_channels)
                output = batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            inputs = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(inputs)
            output = deconv(rectified, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

        return layers[-1]

    def discriminator(self,discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []
        ndf = 64
        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(inputs, ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    def create_model(self,inputs, targets):

        # gan generator
        with tf.variable_scope("generator"):
            out_channels = int(targets.get_shape()[-1])
            outputs = self.generator(inputs, out_channels)
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real = self.discriminator(inputs, targets)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = self.discriminator(inputs, outputs)

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))

            content_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            gen_loss = gen_loss_GAN * self.gan_weight + content_loss_L1 * self.l1_weight

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(self.adam_lr,self.adam_beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(self.adam_lr,self.adam_beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, content_loss_L1])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)

        return Model(
            predict_real=predict_real,
            predict_fake=predict_fake,
            discrim_loss=ema.average(discrim_loss),
            discrim_grads_and_vars=discrim_grads_and_vars,
            gen_loss_GAN=ema.average(gen_loss_GAN),
            content_loss_L1=ema.average(content_loss_L1),
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=outputs,
            train=tf.group(update_losses, incr_global_step, gen_train),
        )

    def train(self):
        examples = self.load_ClearHaze_img()
        print("examples count = %d" % examples.count)

        model = self.create_model(examples.inputs, examples.targets)

        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

        def convert(image):
            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

        with tf.name_scope("convert"):
            with tf.name_scope("inputs"):
                converted_inputs = convert(inputs)
            with tf.name_scope("targets"):
                converted_targets = convert(targets)
            with tf.name_scope("outputs"):
                converted_outputs = convert(outputs)

        # summaries
        with tf.name_scope('summary'):
            with tf.name_scope("img"):
                tf.summary.image("inputs", converted_inputs)
                tf.summary.image("targets", converted_targets)
                tf.summary.image("outputs", converted_outputs)

            with tf.name_scope("predict"):
                tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))
                tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

            tf.summary.scalar("discriminator_loss", model.discrim_loss)
            tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
            tf.summary.scalar("content_loss_L1", model.content_loss_L1)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        def save_images(fetches, step):
            h_ = self.im_size
            w_ = h_*3
            for i in range(0,self.batch_size):
                result = np.ones([ h_,w_ + 10,3], dtype=np.uint8) * 255
                result_name = self.display_dir + "/" + \
                              str(int((time.time() - start) / 60)) + "m_" \
                              + str(step) +"_"+str(i)  + ".png"
                x=0
                for kind in ['inputs','outputs','targets']:
                    tmp = np.array(fetches[kind][i], dtype=np.uint8)
                    result[ 0:,x:x+self.im_size,0:]=tmp
                    x=x+self.im_size+5
                misc.imsave(result_name, result)
            return

        saver = tf.train.Saver(max_to_keep=1)
        logdir = self.log_dir
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        # start tf work
        with sv.managed_session() as sess:
            # print parameter——count
            print("parameter_count =", sess.run(parameter_count))

            max_steps = 2 ** 32
            if self.max_epoch is not None:
                max_steps = examples.steps_per_epoch * self.max_epoch
            print("max steps is {}".format(max_steps))

            display_fetches = {
                "inputs": converted_inputs,
                "targets": converted_targets,
                "outputs": converted_outputs,
            }
            # training
            start = time.time()
            print('star to train')
            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(self.freq_process):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["content_loss_L1"] = model.content_loss_L1

                if should(self.freq_summary):
                    fetches["summary"] = sv.summary_op

                if should(self.freq_display):
                    fetches["display"] = display_fetches
                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(self.freq_summary):
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(self.freq_display):
                    print("saving display images")
                    save_images(results["display"], step=results["global_step"])

                if should(self.freq_trace):
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(self.freq_process):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * self.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * self.batch_size / rate
                    print(
                    "epoch:%d;step:%d;discrim_loss:%0.6f;gen_loss_GAN:%0.6f;content_loss_L1:%0.6f;%0.1f image/sec:remaining:%dm;"
                    % (train_epoch, train_step,
                       results["discrim_loss"], results["gen_loss_GAN"], results["content_loss_L1"],
                       rate, remaining / 60))

                if should(self.freq_save):
                    print("saving model")
                    saver.save(sess, os.path.join(self.ckt_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break
        self.have_trained = True




    def test(self, input_dir, result_dir):
        examples = self.load_ClearHaze_img(istesting=True)
        print("examples count = %d" % examples.count)

        model = self.create_model(examples.inputs, examples.targets)
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

        with tf.name_scope("fetches"):
            def convert(image):
                return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
            converted_inputs = convert(inputs)
            converted_targets = convert(targets)
            converted_outputs = convert(outputs)
            display_fetches = {
                "paths":examples.paths,
                "inputs": converted_inputs,
                "targets": converted_targets,
                "outputs": converted_outputs,
            }

        #   display_fetches = {
        #         "paths": examples.paths,
        #         "inputs": tf.map_fn(tf.image.encode_jpeg, converted_inputs, dtype=tf.string, name="input_jpgs"),
        #         "targets": tf.map_fn(tf.image.encode_jpeg, converted_targets, dtype=tf.string, name="target_jpgs"),
        #         "outputs": tf.map_fn(tf.image.encode_jpeg, converted_outputs, dtype=tf.string, name="output_jpgs"),
        #     }
        # def save_images(fetches, step, image_dir):
        #
        #     for i, in_path in enumerate(fetches["inputs"]):
        #         for kind in ["inputs", "outputs", "targets"]:
        #             name = fetches["paths"][i]
        #             _, name = os.path.split(name)
        #             filename = name + "-" + kind + ".png"
        #             if step is not None:
        #                 filename = "%08d-%s" % (step, filename)
        #             out_path = os.path.join(image_dir, filename)
        #             contents = None
        #             try:
        #                 contents = fetches[kind][i]
        #             except:
        #                 print('##################error%d' % i)
        #             with open(out_path, "wb") as f:
        #                 f.write(contents)
        #     return

        def save_images(fetches):
            h_ = self.im_size
            w_ = h_*3
            for i in range(0,self.batch_size):
                result = np.ones([ h_,w_ + 10,3], dtype=np.uint8) * 255
                _,result_name = os.path.split(str(fetches['paths'][i][0]))
                print result_name
                s=raw_input()
                result_name = result_dir + "/" + result_name
                x=0
                for kind in ['inputs','outputs','targets']:
                    tmp = np.array(fetches[kind][i], dtype=np.uint8)
                    result[ 0:,x:x+self.im_size,0:]=tmp
                    x=x+self.im_size+5
                misc.imsave(result_name, result)
            return

        saver = tf.train.Saver(max_to_keep=2)
        logdir = self.log_dir
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        # start tf work
        if self.have_trained:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.95
            with sv.managed_session(config=config) as sess:
                print("loading model from checkpoint")
                checkpoint = tf.train.latest_checkpoint(self.ckt_dir)
                saver.restore(sess, checkpoint)
                print("loading done")
                for step in range(0, examples.steps_per_epoch+1):
                    results = sess.run(display_fetches)
                    save_images(results)
        else:
            raise ValueError('have not trained')

    def load_ClearHaze_img(self,istesting=False):
        inputs_dir = self.train_dir
        if istesting:
            inputs_dir =self.test_dir
        if not os.path.exists(inputs_dir):
            raise Exception("inputs_dir or targets_dir does not exist")
        inputs_paths = glob.glob(os.path.join(inputs_dir, "*.png"))
        decode = tf.image.decode_png

        # load images
        with tf.name_scope("load_images"):
            path_queue = tf.train.string_input_producer(inputs_paths, shuffle=False)
            reader = tf.WholeFileReader()
            in_paths, contents = reader.read(path_queue)
            raw_input_ = decode(contents)
            raw_inputs = tf.image.convert_image_dtype(raw_input_, dtype=tf.float32)
            assertion = tf.assert_equal(tf.shape(raw_inputs)[2], 3, message="image does not have 3 channels")
            with tf.control_dependencies([assertion]):
                raw_inputs = tf.identity(raw_inputs)

            raw_inputs.set_shape([512, 1024, 3])
            targets = preprocess(raw_inputs[:, :512, :])
            inputs = preprocess(raw_inputs[:, 512:, :])

            inputs.set_shape([512, 512, 3])
            targets.set_shape([512, 512, 3])
        in_paths, inputs_batch, targets_batch = tf.train.shuffle_batch(
                [in_paths,inputs, targets],
                batch_size=self.batch_size,
                capacity=5000,min_after_dequeue=1000,
                num_threads=1)
        if istesting:
            in_paths, inputs_batch, targets_batch = \
                tf.train.batch([in_paths, inputs, targets],batch_size=self.batch_size)
        steps_per_epoch = int(math.ceil(len(inputs_paths) / self.batch_size))

        return Examples(
            paths=in_paths,
            inputs=inputs_batch,
            targets=targets_batch,
            count=len(inputs_paths),
            steps_per_epoch=steps_per_epoch,
        )











