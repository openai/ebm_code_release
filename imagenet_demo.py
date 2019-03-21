from models import ResNet128
import numpy as np
import os.path as osp
from tensorflow.python.platform import flags
import tensorflow as tf
import imageio


flags.DEFINE_string('logdir', '../cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_steps', 200, 'num of steps for conditional imagenet sampling')
flags.DEFINE_float('step_lr', 170., 'number of steps to run')
flags.DEFINE_integer('batch_size', 16, 'number of steps to run')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('spec_norm', True, 'whether to use spectral normalization in weights in a model')
flags.DEFINE_bool('cclass', True, 'conditional models')
flags.DEFINE_bool('use_attention', False, 'using attention')

FLAGS = flags.FLAGS

def rescale_im(im):
    return np.clip(im * 256, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    model = ResNet128(num_channels=3, num_filters=64)
    X_NOISE = tf.placeholder(shape=(None, 128, 128, 3), dtype=tf.float32)
    LABEL = tf.placeholder(shape=(None, 1000), dtype=tf.float32)

    weights = model.construct_weights("context_0")
    sess = tf.Session()

    x_mod = X_NOISE
    x_mod = x_mod + tf.random_normal(tf.shape(x_mod),
                                     mean=0.0,
                                     stddev=0.005)

    energy_noise = energy_start = tf.concat(
        [model.forward(
                x_mod,
                weights,
                label=LABEL,
                reuse=True,
                stop_at_grad=False,
                stop_batch=True)],
        axis=0)

    x_grad = tf.gradients(energy_noise, [x_mod])[0]
    energy_noise_old = energy_noise

    lr = FLAGS.step_lr

    x_last = x_mod - (lr) * x_grad

    x_mod = x_last
    x_mod = tf.clip_by_value(x_mod, 0, 1)
    x_output = x_mod

    saver = loader = tf.train.Saver(
        max_to_keep=30, keep_checkpoint_every_n_hours=6)

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
    saver.restore(sess, model_file)

    ls = np.random.permutation(1000)[:16]
    ims = []

    # What to initialize sampling with. 
    x_mod = np.random.uniform(0, 1, size=(FLAGS.batch_size, 128, 128, 3))
    labels = np.eye(1000)[lx]

    for i in range(200):
        x_mod = sess.run(x_output, {X_NOISE:x_mod, LABEL:labels})
        ims.append(rescale_im(x_mod).reshape((4, 4, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape((512, 512, 3)))

    imageio.mimwrite(osp.join(logdir, 'sample.gif'), ims)


