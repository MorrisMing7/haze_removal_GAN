import tensorflow as tf  # version=1.4.1
import os
import glob
import collections
import numpy as np
import time
import math
import scipy.misc as misc

Model = collections.namedtuple("Model",
    "g_imgs, d_real, d_fake, g_loss, d_loss, "
    "W_dist, gradient_penalty, g_train, d_train, global_step")

class Data:
    def __init__(self, data_dir, img_size):
        if not os.path.exists(data_dir):
            raise Exception("data_dir doesn't exist")
        self.data_dir = data_dir
        self.index = 0
        self.img_size = img_size
        self.file_paths = glob.glob(os.path.join(data_dir, '*.jpg'))
        self.data = np.zeros([len(self.file_paths), img_size, img_size, 1], dtype=np.float32)
        for i, name in enumerate(self.file_paths):
            tmp = misc.imread(name)
            tmp = np.array(tmp.astype(np.float32), dtype=np.float32)
            self.data[i, :, :, :] = tmp.reshape([1, img_size, img_size, 1])
        #zero-center normalization
        self.data = self.data / 255*2-1

    def __call__(self, batch_size):
        result = np.ones([batch_size, self.img_size, self.img_size, 1], dtype=np.float32)
        if self.index + batch_size > len(self.file_paths):
            #shuffle when reach the end
            np.random.shuffle(self.data)
            self.index=0
        tmp = self.index + batch_size
        result[:, :, :, :] = self.data[self.index:tmp, :, :, :]
        self.index = tmp
        assert result.shape[0] == batch_size
        return result


class CloudGAN:
    def __init__(self, real_img_dir, display_dir, sample_dir, log_dir, ckpt_dir,
                 img_size=256, dim_z=256, batch_size=36,
                 D_lr=5e-4, G_lr=1e-4, beta1=0.0, beta2=0.99,_lambda=10.,
                 #it turned out that we do not need too much steps
                 g_first_layer=(32, 32, 512), max_steps=5000,
                 g_layers=(  # (bs,256) #(bs,32,32,512)
                         # ( (kernal_size,stride,channels)...)
                         (4, 2, 256),  # (bs,64,64,256)
                         (4, 2, 128),  # (bs,128,128,128)
                         (4, 2, 64),  # (bs,256,256,64)
                         (4, 2, 1)),  # (bs,256,256,1)
                 d_layers=(  # (bs,256,256,1)
                         (4, 2, 64),  # (bs,128,128,64)
                         (4, 2, 128),  # (bs,64,64,128)
                         (4, 2, 256),  # (bs,32,32,256)
                         (4, 2, 512)),  # (bs,16,16,512)
                 ):

        self.data = Data(real_img_dir, img_size)
        self.display_dir = display_dir
        self.sample_dir = sample_dir
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir

        self.D_lr = D_lr
        self.G_lr = G_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self._lambda=_lambda

        self.img_size = img_size
        self.dim_z = dim_z
        self.batch_size = batch_size \
            if math.pow(math.floor(math.sqrt(batch_size)),2) - batch_size == 0 else 25
        self.gfl = g_first_layer
        self.max_steps = max_steps
        self.g_layers = g_layers
        self.d_layers = d_layers

    def generator(self, z):
        layers = []
        with tf.control_dependencies([tf.assert_equal(tf.shape(z), [self.batch_size, self.dim_z])]):
            layers.append(z)
        with tf.variable_scope("g_first_dense"):
            gfl = self.gfl
            output = tf.layers.dense(
                layers[-1], units=gfl[0] * gfl[1] * gfl[2], activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.02), use_bias=False)
            output = tf.layers.batch_normalization(output, axis=0)
            output = tf.reshape(output, [self.batch_size] + list(gfl))
            layers.append(output)
        for i, (ks, stride, channels) in enumerate(self.g_layers):
            with tf.variable_scope('g_deconv_layer{}'.format(i)):
                output = tf.layers.conv2d_transpose(
                    layers[-1], filters=channels, kernel_size=ks, padding='same',
                    strides=[stride, stride] if i < len(self.g_layers) - 1 else [1, 1],
                    #use tanh at the last layer
                    activation=tf.nn.relu if i<len(self.g_layers)-1 else tf.nn.tanh,
                    kernel_initializer=tf.random_normal_initializer(0, 0.02)
                )
                if i < len(self.g_layers):
                    output = tf.layers.batch_normalization(output, axis=0)
            layers.append(output)
        assert layers[-1].get_shape().as_list() == [self.batch_size, self.img_size, self.img_size, 1]
        return layers[-1]

    def discriminator(self, img):
        layers = []
        with tf.control_dependencies(
                [tf.assert_equal(tf.shape(img), [self.batch_size, self.img_size, self.img_size, 1])]):
            layers.append(img)
        for i, (ks, stride, channels) in enumerate(self.d_layers):
            with tf.variable_scope('d_conv_layer{}'.format(i)):
                output = tf.layers.conv2d(
                    layers[-1], filters=channels, kernel_size=ks,
                    strides=[stride, stride], padding='same', activation=tf.nn.leaky_relu
                )
            layers.append(output)
        with tf.variable_scope('d_last_dense'):
            output = tf.reshape(layers[-1], [self.batch_size, -1])
            output = tf.layers.dense(output, units=1, activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0, 0.02),
                                     bias_initializer=tf.random_normal_initializer(0, 0.02))
            output = tf.reshape(output, [self.batch_size, 1, 1, 1])
            layers.append(output)

        return layers[-1]

    def create_model(self, real_imgs, z_noise):

        with tf.variable_scope("generator"):
            fake_imgs = self.generator(z_noise)

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            d_real = self.discriminator(real_imgs)
            d_fake = self.discriminator(fake_imgs)

        with tf.name_scope("loss"):
            W_dist = tf.reduce_mean(d_real - d_fake)
            d_loss = -W_dist
            g_loss = tf.reduce_mean(-d_fake)
        eps = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.2, maxval=0.8)
        x_hat = eps * real_imgs + (1. - eps) * fake_imgs

        with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
            d_xhat = self.discriminator(x_hat)

        with tf.name_scope("loss"):
            d_xhat_grad = tf.gradients(d_xhat, x_hat)[0]  # gradient of D(x_hat)
            d_xhat_grad_norm = tf.norm(tf.layers.flatten(d_xhat_grad), axis=1)  # l2 norm
            GP = self._lambda * tf.reduce_mean(tf.square(d_xhat_grad_norm - 1.))
            d_loss += GP

        with tf.name_scope("train"):
            d_train = tf.train.AdamOptimizer(learning_rate=self.D_lr,
                                                beta1=self.beta1, beta2=self.beta2).\
                minimize(loss=d_loss,
                        var_list=[var for var in tf.trainable_variables()
                          if var.name.startswith("discriminator")] )
            global_step = tf.Variable(0, name='global_step', trainable=False)
            g_train = tf.train.AdamOptimizer(learning_rate=self.G_lr,
                                                beta1=self.beta1, beta2=self.beta2)\
                .minimize(loss=g_loss,
                        var_list=[var for var in tf.trainable_variables()
                          if var.name.startswith('generator')],global_step=global_step  )

        return Model(g_imgs=fake_imgs,d_real=d_real,d_fake=d_fake,
                     g_loss=g_loss,d_loss=d_loss,
                     W_dist=W_dist,gradient_penalty=GP,
                     g_train=g_train,d_train=d_train,global_step=global_step)

    def train(self):
        real_imgs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.img_size, self.img_size, 1])
        z_noise = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.dim_z])
        model = self.create_model(real_imgs, z_noise)
        with tf.name_scope("summary"):
            with tf.name_scope("loss"):
                tf.summary.scalar("g_loss", model.g_loss)
                tf.summary.scalar("d_loss", model.d_loss)
            with tf.name_scope("real"):
                tf.summary.image("img", tf.image.convert_image_dtype((real_imgs+1)/2, dtype=tf.uint8, saturate=True))
                tf.summary.image("predict",
                                 tf.image.convert_image_dtype(model.d_real, dtype=tf.uint8, saturate=False))
            with tf.name_scope("fake"):
                tf.summary.image("img", tf.image.convert_image_dtype((model.g_imgs+1)/2, dtype=tf.uint8, saturate=True))
                tf.summary.image("predict",
                                 tf.image.convert_image_dtype(model.d_fake, dtype=tf.uint8, saturate=False))

        saver = tf.train.Saver(max_to_keep=3)
        sv = tf.train.Supervisor(logdir=self.log_dir, save_summaries_secs=0, saver=None)

        def sample_z(m, n):
            return np.random.uniform(low=-1, high=1, size=[m, n])

        def save_images(imgs, step):
            hw = int(math.sqrt(self.batch_size))
            hw_ = self.img_size * hw + 5 * (hw - 1)
            result = np.ones([hw_, hw_], dtype=np.uint8) * 255
            result_name = self.display_dir + "/" + \
                          str(int((time.time() - train_start_time) / 60)) + "m_" + str(step) + ".png"
            for i in range(hw):
                for j in range(hw):
                    tmp = (np.array(imgs[i * hw + j], dtype=np.float32)+1)/2 * 255
                    tmp = tmp.astype(dtype=np.uint8)
                    tmp = tmp.reshape([self.img_size, self.img_size])
                    x = i * (self.img_size + 5)
                    y = j * (self.img_size + 5)
                    result[x:x + self.img_size, y:y + self.img_size] = tmp
            misc.imsave(result_name, result)

        train_start_time = time.time()
        with sv.managed_session() as sess:
            for step in range(self.max_steps):
                ri=self.data(self.batch_size)
                for i in range(3):
                    sess.run([model.d_train],{real_imgs:ri,
                                z_noise:sample_z(self.batch_size,self.dim_z)})
                    _,global_step = sess.run([model.g_train,model.global_step],
                                {z_noise:sample_z(self.batch_size,self.dim_z)})
                fake_imgs, d_loss, g_loss, summary = sess.run(
                    [model.g_imgs, model.d_loss, model.g_loss, sv.summary_op],
                    feed_dict={real_imgs: ri, z_noise: sample_z(self.batch_size, self.dim_z)})
                if step%100==1 or step<100 and step%5==0:
                    save_images(fake_imgs, step)
                print("step{:0>4d} d_loss:{:6e} g_loss:{:6e}".format(step, d_loss, g_loss) +
                      " using:{}m; {:.2f}sec/step ".format(
                          int((time.time() - train_start_time) / 60),
                          (time.time() - train_start_time) / (step + 1) ))
                sv.summary_writer.add_summary(summary, global_step=global_step)
                if step%1000==0 or step==self.max_steps-1:
                    saver.save(sess,save_path=os.path.join(self.ckpt_dir,'model'),global_step=global_step)

                if sv.should_stop():
                    break
        return

    def test(self,num_batches):
        real_imgs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.img_size, self.img_size, 1])
        z_noise = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.dim_z])
        model = self.create_model(real_imgs, z_noise)
        saver = tf.train.Saver(max_to_keep=3)
        sv = tf.train.Supervisor(logdir=self.log_dir,save_summaries_secs=0,saver=None)
        def sample_z(m, n):
            return np.random.uniform(low=-1, high=1, size=[m, n])
        def save_images(imgs, step):
            result_name = self.sample_dir + "/"  + str(step)+"_"
            for i in range(self.batch_size):
                tmp = (np.array(imgs[i], dtype=np.float32)+1)/2 * 255
                tmp = tmp.astype(dtype=np.uint8)
                tmp = tmp.reshape([self.img_size, self.img_size])
                name = result_name +str(i)+".png"
                misc.imsave(name, tmp)
        with sv.managed_session() as sess:
            print('loading model from checkpoint')
            ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
            ri = self.data(self.batch_size)
            saver.restore(sess,ckpt)
            print('producing images...')
            for step in range(num_batches):
                fake_imgs = sess.run(model.g_imgs,
                    feed_dict={real_imgs:ri,z_noise:sample_z(self.batch_size,self.dim_z)})
                save_images(fake_imgs,step)
                if (step+1)%50==0:
                    print("{} images done".format((step+1)*self.batch_size))
            print('\ncomplete, {} images done'.format(num_batches*self.batch_size))
        return

