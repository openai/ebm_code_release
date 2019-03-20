import tensorflow as tf
import math
from hmc import hmc
from tensorflow.python.platform import flags
from torch.utils.data import DataLoader
from models import DspritesNet, ResNet32, ResNet32Large, ResNet32Wider, MnistNet
from data import Cifar10, Mnist, DSprites
from scipy.misc import logsumexp
from scipy.misc import imsave
from utils import optimistic_restore
import os.path as osp
import numpy as np
from tqdm import tqdm

flags.DEFINE_string('datasource', 'random', 'default or noise or negative or single')
flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or mnist or dsprites or 2d or toy Gauss')
flags.DEFINE_string('logdir', '/mnt/nfs/yilundu/ebm_code_release/cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('data_workers', 5, 'Number of different data workers to load data in parallel')
flags.DEFINE_integer('batch_size', 16, 'Size of inputs')
flags.DEFINE_string('resume_iter', '-1', 'iteration to resume training from')

flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omniglot.')
flags.DEFINE_integer('pdist', 10, 'number of intermediate distributions for ais')
flags.DEFINE_integer('gauss_dim', 500, 'dimensions for modeling Gaussian')
flags.DEFINE_integer('rescale', 1, 'factor to rescale input outside of normal (0, 1) box')
flags.DEFINE_float('temperature', 1, 'temperature at which to compute likelihood of model')
flags.DEFINE_bool('bn', False, 'Whether to use batch normalization or not')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_bias', True, 'Whether to use bias in convolution')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_bool('cclass', False, 'Whether to evaluate the log likelihood of conditional model or not')
flags.DEFINE_bool('single', False, 'Whether to evaluate the log likelihood of conditional model or not')
flags.DEFINE_bool('large_model', False, 'Use large model to evaluate')
flags.DEFINE_bool('wider_model', False, 'Use large model to evaluate')
flags.DEFINE_float('alr', 0.0045, 'Learning rate to use for HMC steps')

FLAGS = flags.FLAGS

label_default = np.eye(10)[0:1, :]
label_default = tf.Variable(tf.convert_to_tensor(label_default, np.float32))


def unscale_im(im):
    return (255 * np.clip(im, 0, 1)).astype(np.uint8)

def gauss_prob_log(x, prec=1.0):

    nh = float(np.prod([s.value for s in x.get_shape()[1:]]))
    norm_constant_log = -0.5 * (tf.log(2 * math.pi) * nh - nh * tf.log(prec))
    prob_density_log = -tf.reduce_sum(tf.square(x - 0.5), axis=[1]) / 2. * prec

    return norm_constant_log + prob_density_log


def uniform_prob_log(x):

    return tf.zeros(1)


def model_prob_log(x, e_func, weights, temp):
    if FLAGS.cclass:
        batch_size = tf.shape(x)[0]
        label_tiled = tf.tile(label_default, (batch_size, 1))
        e_raw = e_func.forward(x, weights, label=label_tiled)
    else:
        e_raw = e_func.forward(x, weights)
    energy = tf.reduce_sum(e_raw, axis=[1])
    return -temp * energy


def bridge_prob_neg_log(alpha, x, e_func, weights, temp):

    if FLAGS.dataset == "gauss":
        norm_prob =  (1-alpha) * uniform_prob_log(x) + alpha * gauss_prob_log(x, prec=FLAGS.temperature)
    else:
        norm_prob =  (1-alpha) * uniform_prob_log(x) + alpha * model_prob_log(x, e_func, weights, temp)
    # Add an additional log likelihood penalty so that points outside of (0, 1) box are *highly* unlikely


    if FLAGS.dataset == '2d' or FLAGS.dataset == 'gauss':
        oob_prob = tf.reduce_sum(tf.square(100 * (x - tf.clip_by_value(x, 0, FLAGS.rescale))), axis = [1])
    elif FLAGS.dataset == 'mnist':
        oob_prob = tf.reduce_sum(tf.square(100 * (x - tf.clip_by_value(x, 0, FLAGS.rescale))), axis = [1, 2])
    else:
        oob_prob = tf.reduce_sum(tf.square(100 * (x - tf.clip_by_value(x, 0., FLAGS.rescale))), axis = [1, 2, 3])

    return -norm_prob + oob_prob


def ancestral_sample(e_func, weights, batch_size=128, prop_dist=10, temp=1, hmc_step=10):
    if FLAGS.dataset == "2d":
        x = tf.placeholder(tf.float32, shape=(None, 2))
    elif FLAGS.dataset == "gauss":
        x = tf.placeholder(tf.float32, shape=(None, FLAGS.gauss_dim))
    elif FLAGS.dataset == "mnist":
        x = tf.placeholder(tf.float32, shape=(None, 28, 28))
    else:
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

    x_init = x

    alpha_prev = tf.placeholder(tf.float32, shape=())
    alpha_new = tf.placeholder(tf.float32, shape=())
    approx_lr = tf.placeholder(tf.float32, shape=())

    chain_weights = tf.zeros(batch_size)
    # for i in range(1, prop_dist+1):
    #     print("processing loop {}".format(i))
    #     alpha_prev = (i-1) / prop_dist
    #     alpha_new = i / prop_dist

    prob_log_old_neg = bridge_prob_neg_log(alpha_prev, x, e_func, weights, temp)
    prob_log_new_neg = bridge_prob_neg_log(alpha_new, x, e_func, weights, temp)

    chain_weights = -prob_log_new_neg + prob_log_old_neg
    # chain_weights = tf.Print(chain_weights, [chain_weights])

    # Sample new x using HMC
    def unorm_prob(x):
        return bridge_prob_neg_log(alpha_new, x, e_func, weights, temp)

    for j in range(1):
        x = hmc(x, approx_lr, hmc_step, unorm_prob)

    return chain_weights, alpha_prev, alpha_new, x, x_init, approx_lr


def main():

    # Initialize dataset
    if FLAGS.dataset == 'cifar10':
        dataset = Cifar10(train=False, rescale=FLAGS.rescale)
        channel_num = 3
        dim_input = 32 * 32 * 3
    elif FLAGS.dataset == 'imagenet':
        dataset = ImagenetClass()
        channel_num = 3
        dim_input = 64 * 64 * 3
    elif FLAGS.dataset == 'mnist':
        dataset = Mnist(train=False, rescale=FLAGS.rescale)
        channel_num = 1
        dim_input = 28 * 28 * 1
    elif FLAGS.dataset == 'dsprites':
        dataset = DSprites()
        channel_num = 1
        dim_input = 64 * 64 * 1
    elif FLAGS.dataset == '2d' or FLAGS.dataset == 'gauss':
        dataset = Box2D()

    dim_output = 1
    data_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, drop_last=False, shuffle=True)

    if FLAGS.dataset == 'mnist':
        model = MnistNet(num_channels=channel_num)
    elif FLAGS.dataset == 'cifar10':
        if FLAGS.large_model:
            model = ResNet32Large(num_filters=128)
        elif FLAGS.wider_model:
            model = ResNet32Wider(num_filters=192)
        else:
            model = ResNet32(num_channels=channel_num, num_filters=128)
    elif FLAGS.dataset == 'dsprites':
        model = DspritesNet(num_channels=channel_num, num_filters=FLAGS.num_filters)

    weights = model.construct_weights('context_{}'.format(0))

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    saver = loader = tf.train.Saver(max_to_keep=10)

    sess.run(tf.global_variables_initializer())
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
    resume_itr = FLAGS.resume_iter

    if FLAGS.resume_iter != "-1":
        optimistic_restore(sess, model_file)
    else:
        print("WARNING, YOU ARE NOT LOADING A SAVE FILE")
    # saver.restore(sess, model_file)

    chain_weights, a_prev, a_new, x, x_init, approx_lr = ancestral_sample(model, weights, FLAGS.batch_size, temp=FLAGS.temperature)
    print("Finished constructing ancestral sample ...................")

    if FLAGS.dataset != "gauss":
        comb_weights_cum = []
        batch_size = tf.shape(x_init)[0]
        label_tiled = tf.tile(label_default, (batch_size, 1))
        e_compute = -FLAGS.temperature * model.forward(x_init, weights, label=label_tiled)
        e_pos_list = []

        for data_corrupt, data, label_gt in tqdm(data_loader):
            e_pos = sess.run([e_compute], {x_init: data})[0]
            e_pos_list.extend(list(e_pos))

        print(len(e_pos_list))
        print("Positive sample probability ", np.mean(e_pos_list), np.std(e_pos_list))

    if FLAGS.dataset == "2d":
        alr = 0.0045
    elif FLAGS.dataset == "gauss":
        alr = 0.0085
    elif FLAGS.dataset == "mnist":
        alr = 0.0065
        #90 alr = 0.0035
    else:
        # alr = 0.0125
        if FLAGS.rescale == 8:
            alr = 0.0085
        else:
            alr = 0.0045
# 
    for i in range(1):
        tot_weight = 0
        for j in tqdm(range(1, FLAGS.pdist+1)):
            if j == 1:
                if FLAGS.dataset == "cifar10":
                    x_curr =  np.random.uniform(0, FLAGS.rescale, size=(FLAGS.batch_size, 32, 32, 3))
                elif FLAGS.dataset == "gauss":
                    x_curr =  np.random.uniform(0, FLAGS.rescale, size=(FLAGS.batch_size, FLAGS.gauss_dim))
                elif FLAGS.dataset == "mnist":
                    x_curr =  np.random.uniform(0, FLAGS.rescale, size=(FLAGS.batch_size, 28, 28))
                else:
                    x_curr =  np.random.uniform(0, FLAGS.rescale, size=(FLAGS.batch_size, 2))

            alpha_prev = (j-1) / FLAGS.pdist
            alpha_new = j / FLAGS.pdist
            cweight, x_curr = sess.run([chain_weights, x], {a_prev: alpha_prev, a_new: alpha_new, x_init: x_curr, approx_lr: alr * (5 ** (2.5*-alpha_prev))})
            tot_weight = tot_weight + cweight

        print("Total values of lower value based off forward sampling", np.mean(tot_weight), np.std(tot_weight))

        tot_weight = 0

        for j in tqdm(range(FLAGS.pdist, 0, -1)):
            alpha_new = (j-1) / FLAGS.pdist
            alpha_prev = j / FLAGS.pdist
            cweight, x_curr = sess.run([chain_weights, x], {a_prev: alpha_prev, a_new: alpha_new, x_init: x_curr, approx_lr: alr * (5 ** (2.5*-alpha_prev))})
            tot_weight = tot_weight - cweight

        print("Total values of upper value based off backward sampling", np.mean(tot_weight), np.std(tot_weight))



if __name__ == "__main__":
    main()
