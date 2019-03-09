import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags

from data import Imagenet, Cifar10, DSprites, Mnist
from models import DspritesNet, ResNet32, ResNet32Large, ResNet32Larger, ResNet32Wider, MnistNet
import os.path as osp
import os
from baselines.logger import TensorBoardOutputFormat
from utils import average_gradients, ReplayBuffer, optimistic_restore
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import time as time
from io import StringIO
from tensorflow.core.util import event_pb2
import torch
import numpy as np
from custom_adam import AdamOptimizer
from scipy.misc import imsave
import matplotlib.pyplot as plt
from hmc import hmc

import horovod.tensorflow as hvd
hvd.init()

from inception import get_inception_score

torch.manual_seed(hvd.rank())
np.random.seed(hvd.rank())
tf.set_random_seed(hvd.rank())

FLAGS = flags.FLAGS


# Dataset Options
flags.DEFINE_string('datasource', 'random',
    'initialization for chains, either random or default (decorruption)')
flags.DEFINE_string('dataset','omniglot',
    'omniglot or imagenet or omniglotfull or cifar10 or mnist or dsprites or 2d')
flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_bool('single', False, 'whether to debug by training on a single image')
flags.DEFINE_integer('data_workers', 4,
    'Number of different data workers to load data in parallel')

# General Experiment Settings
flags.DEFINE_string('logdir', '/mnt/nfs/yilundu/ebm_code_release/cachedir',
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 1000,'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000,'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_integer('epoch_num', 10000, 'Number of Epochs to train on')
flags.DEFINE_float('lr', 3e-4, 'Learning for training')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus to train on')

# EBM Specific Experiments Settings
flags.DEFINE_float('ml_coeff', 1.0, 'Maximum Likelihood Coefficients')
flags.DEFINE_bool('cclass', False, 'Whether to conditional training in models')
flags.DEFINE_bool('model_cclass', False,'use unsupervised clustering to infer fake labels')
flags.DEFINE_integer('temperature', 1, 'Temperature for energy function')
flags.DEFINE_string('objective', 'cd', 'use either contrastive divergence objective(least stable),'
                    'logsumexp(more stable)'
                    'softplus(most stable)')
flags.DEFINE_bool('zero_kl', False, 'whether to zero out the kl loss')

# Setting for MCMC sampling
flags.DEFINE_float('proj_norm', 0.0, 'Maximum change of input images')
flags.DEFINE_string('proj_norm_type', 'li', 'Either li or l2 ball projection')
flags.DEFINE_integer('num_steps', 20, 'Steps of gradient descent for training')
flags.DEFINE_float('step_lr', 1.0, 'Size of steps for gradient descent')
flags.DEFINE_bool('replay_batch', False, 'Use MCMC chains initialized from a replay buffer.')
flags.DEFINE_bool('hmc', False, 'Whether to use HMC sampling to train models')
flags.DEFINE_float('noise_scale', 1.,'Relative amount of noise for MCMC')

# Architecture Settings
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_bool('large_model', False, 'whether to use a large model')
flags.DEFINE_bool('larger_model', False, 'Deeper ResNet32 Network')
flags.DEFINE_bool('wider_model', False, 'Wider ResNet32 Network')

# Dataset settings
flags.DEFINE_bool('mixup', False, 'whether to add mixup to training images')
flags.DEFINE_bool('augment', False, 'whether to augmentations to images')
flags.DEFINE_float('rescale', 1.0, 'Factor to rescale inputs from 0-1 box')

# Dsprites specific experiments
flags.DEFINE_bool('cond_shape', False, 'condition of shape type')
flags.DEFINE_bool('cond_size', False, 'condition of shape size')
flags.DEFINE_bool('cond_pos', False, 'condition of position loc')
flags.DEFINE_bool('cond_rot', False, 'condition of rot')

FLAGS.step_lr = FLAGS.step_lr * FLAGS.rescale

FLAGS.batch_size *= FLAGS.num_gpus

print("{} batch size".format(FLAGS.batch_size))


def compress_x_mod(x_mod):
    x_mod = (256 * np.clip(x_mod, 0, FLAGS.rescale) / FLAGS.rescale).astype(np.uint8)
    return x_mod


def decompress_x_mod(x_mod):
    x_mod = x_mod / 256 * FLAGS.rescale + \
        np.random.uniform(0, 1 / 256 * FLAGS.rescale, x_mod.shape)
    return x_mod


def make_image(tensor):
    """Convert an numpy representation image to Image protobuf"""
    from PIL import Image
    if len(tensor.shape) == 4:
        _, height, width, channel = tensor.shape
    elif len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    elif len(tensor.shape) == 2:
        height, width = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


def log_image(im, logger, tag, step=0):
    im = make_image(im)

    summary = [tf.Summary.Value(tag=tag, image=im)]
    summary = tf.Summary(value=summary)
    event = event_pb2.Event(summary=summary)
    event.step = step
    logger.writer.WriteEvent(event)
    logger.writer.Flush()


def rescale_im(image):
    image = np.clip(image, 0, FLAGS.rescale)
    if FLAGS.dataset == 'omniglot' or FLAGS.dataset == 'omniglotfull' or FLAGS.dataset == 'mnist' or FLAGS.dataset == 'dsprites':
        return ((FLAGS.rescale - image) * 256 / FLAGS.rescale).astype(np.uint8)
    else:
        return (image * 256 / FLAGS.rescale).astype(np.uint8)


def train(target_vars, saver, sess, logger, dataloader, resume_iter, logdir):
    X = target_vars['X']
    Y = target_vars['Y']
    X_NOISE = target_vars['X_NOISE']
    train_op = target_vars['train_op']
    energy_pos = target_vars['energy_pos']
    energy_neg = target_vars['energy_neg']
    loss_energy = target_vars['loss_energy']
    loss_ml = target_vars['loss_ml']
    loss_total = target_vars['total_loss']
    gvs = target_vars['gvs']
    x_grad = target_vars['x_grad']
    x_grad_first = target_vars['x_grad_first']
    x_off = target_vars['x_off']
    temp = target_vars['temp']
    x_mod = target_vars['x_mod']
    LABEL = target_vars['LABEL']
    LABEL_POS = target_vars['LABEL_POS']
    weights = target_vars['weights']
    test_x_mod = target_vars['test_x_mod']
    eps = target_vars['eps_begin']
    label_ent = target_vars['label_ent']

    if FLAGS.use_attention:
        gamma = weights[0]['atten']['gamma']
    else:
        gamma = tf.zeros(1)

    val_output = [test_x_mod]

    gvs_dict = dict(gvs)

    log_output = [
        train_op,
        energy_pos,
        energy_neg,
        eps,
        loss_energy,
        loss_ml,
        loss_total,
        x_grad,
        x_off,
        x_mod,
        gamma,
        x_grad_first,
        label_ent,
        *gvs_dict.keys()]
    output = [train_op, x_mod]

    replay_buffer = ReplayBuffer(50000)
    itr = resume_iter
    x_mod = None
    gd_steps = 1

    dataloader_iterator = iter(dataloader)
    best_inception = 0.0

    for epoch in range(FLAGS.epoch_num):
        for data_corrupt, data, label in dataloader:
            data_corrupt = data_corrupt_init = data_corrupt.numpy()
            data_corrupt_init = data_corrupt.copy()

            data = data.numpy()
            label = label.numpy()

            label_init = label.copy()

            if FLAGS.mixup:
                idx = np.random.permutation(data.shape[0])
                lam = np.random.beta(1, 1, size=(data.shape[0], 1, 1, 1))
                data = data * lam + data[idx] * (1 - lam)

            if FLAGS.replay_batch and (x_mod is not None):
                replay_buffer.add(compress_x_mod(x_mod))

                if len(replay_buffer) > FLAGS.batch_size:
                    replay_batch = replay_buffer.sample(FLAGS.batch_size)
                    replay_batch = decompress_x_mod(replay_batch)
                    replay_mask = (
                        np.random.uniform(
                            0,
                            FLAGS.rescale,
                            FLAGS.batch_size) > 0.05)
                    data_corrupt[replay_mask] = replay_batch[replay_mask]

            feed_dict = {X_NOISE: data_corrupt, X: data, Y: label}

            if FLAGS.cclass:
                feed_dict[LABEL] = label
                feed_dict[LABEL_POS] = label_init

            if itr % FLAGS.log_interval == 0:
                _, e_pos, e_neg, eps, loss_e, loss_ml, loss_total, x_grad, x_off, x_mod, gamma, x_grad_first, label_ent, * \
                    grads = sess.run(log_output, feed_dict)

                kvs = {}
                kvs['e_pos'] = e_pos.mean()
                kvs['e_pos_std'] = e_pos.std()
                kvs['e_neg'] = e_neg.mean()
                kvs['e_diff'] = kvs['e_pos'] - kvs['e_neg']
                kvs['e_neg_std'] = e_neg.std()
                kvs['temp'] = temp
                kvs['loss_e'] = loss_e.mean()
                kvs['eps'] = eps.mean()
                kvs['label_ent'] = label_ent
                kvs['loss_ml'] = loss_ml.mean()
                kvs['loss_total'] = loss_total.mean()
                kvs['x_grad'] = np.abs(x_grad).mean()
                kvs['x_grad_first'] = np.abs(x_grad_first).mean()
                kvs['x_off'] = x_off.mean()
                kvs['iter'] = itr
                kvs['gamma'] = gamma

                for v, k in zip(grads, [v.name for v in gvs_dict.values()]):
                    kvs[k] = np.abs(v).max()

                string = "Obtained a total of "
                for key, value in kvs.items():
                    string += "{}: {}, ".format(key, value)

                if hvd.rank() == 0:
                    print(string)
                    logger.writekvs(kvs)
                else:
                    _, x_mod = sess.run(output, feed_dict)

                if itr % FLAGS.save_interval == 0 and hvd.rank() == 0:
                    saver.save(
                        sess,
                        osp.join(
                            FLAGS.logdir,
                            FLAGS.exp,
                            'model_{}'.format(itr)))

                if itr % FLAGS.test_interval == 0 and hvd.rank() == 0 and FLAGS.dataset != '2d':
                    try_im = x_mod
                    orig_im = data_corrupt.squeeze()
                    actual_im = rescale_im(data)

                    orig_im = rescale_im(orig_im)
                    try_im = rescale_im(try_im).squeeze()

                    for i, (im, t_im, actual_im_i) in enumerate(
                            zip(orig_im[:20], try_im[:20], actual_im)):
                        shape = orig_im.shape[1:]
                        new_im = np.zeros((shape[0], shape[1] * 3, *shape[2:]))
                        size = shape[1]
                        new_im[:, :size] = im
                        new_im[:, size:2 * size] = t_im
                        new_im[:, 2 * size:] = actual_im_i

                        log_image(
                            new_im, logger, 'train_gen_{}'.format(itr), step=i)

                    test_im = x_mod

                    try:
                        data_corrupt, data, label = next(dataloader_iterator)
                    except BaseException:
                        dataloader_iterator = iter(dataloader)
                        data_corrupt, data, label = next(dataloader_iterator)

                    data_corrupt = data_corrupt.numpy()

                    if FLAGS.replay_batch and (
                            x_mod is not None) and len(replay_buffer) > 0:
                        replay_batch = replay_buffer.sample(FLAGS.batch_size)
                        replay_batch = decompress_x_mod(replay_batch)
                        replay_mask = (
                            np.random.uniform(
                                0, 1, (FLAGS.batch_size)) > 0.05)
                        data_corrupt[replay_mask] = replay_batch[replay_mask]

                    if FLAGS.dataset == 'cifar10' or FLAGS.dataset == 'imagenet':
                        if len(replay_buffer) > 128:
                            feed_dict[X_NOISE] = decompress_x_mod(
                                replay_buffer.sample(128))
                        else:
                            feed_dict[X_NOISE] = np.random.uniform(
                                0, FLAGS.rescale, (128, 32, 32, 3))

                        if FLAGS.dataset == 'cifar10':
                            label = np.eye(10)[np.random.randint(0, 10, (128))]
                        else:
                            label = np.eye(1000)[
                                np.random.randint(
                                    0, 1000, (128))]
                    else:
                        feed_dict[X_NOISE] = data_corrupt

                    feed_dict[X] = data

                    if FLAGS.cclass:
                        feed_dict[LABEL] = label

                    test_x_mod = sess.run(val_output, feed_dict)

                    try_im = test_x_mod
                    orig_im = data_corrupt.squeeze()
                    actual_im = rescale_im(data.numpy())

                    orig_im = rescale_im(orig_im)
                    try_im = rescale_im(try_im).squeeze()

                    for i, (im, t_im, actual_im_i) in enumerate(
                            zip(orig_im[:20], try_im[:20], actual_im)):

                        shape = orig_im.shape[1:]
                        new_im = np.zeros((shape[0], shape[1] * 3, *shape[2:]))
                        size = shape[1]
                        new_im[:, :size] = im
                        new_im[:, size:2 * size] = t_im
                        new_im[:, 2 * size:] = actual_im_i
                        log_image(
                            new_im, logger, 'val_gen_{}'.format(itr), step=i)

                    score, std = get_inception_score(list(try_im))
                    print(
                        "Inception score of {} with std of {}".format(
                            score, std))
                    kvs = {}
                    kvs['inception_score'] = score
                    kvs['inception_score_std'] = std
                    logger.writekvs(kvs)

                    if score > best_inception:
                        best_inception = score
                        saver.save(
                            sess,
                            osp.join(
                                FLAGS.logdir,
                                FLAGS.exp,
                                'model_best'))

            itr += 1

    saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))


cifar10_map = {0: 'airplane',
               1: 'automobile',
               2: 'bird',
               3: 'cat',
               4: 'deer',
               5: 'dog',
               6: 'frog',
               7: 'horse',
               8: 'ship',
               9: 'truck'}


def test(target_vars, saver, sess, logger, dataloader):
    X_NOISE = target_vars['X_NOISE']
    X = target_vars['X']
    Y = target_vars['Y']
    LABEL = target_vars['LABEL']
    energy_start = target_vars['energy_start']
    x_mod = target_vars['x_mod']
    x_mod = target_vars['test_x_mod']
    energy_neg = target_vars['energy_neg']

    np.random.seed(1)
    random.seed(1)

    output = [x_mod, energy_start, energy_neg]

    dataloader_iterator = iter(dataloader)
    data_corrupt, data, label = next(dataloader_iterator)
    data_corrupt, data, label = data_corrupt.numpy(), data.numpy(), label.numpy()

    orig_im = try_im = data_corrupt

    if FLAGS.cclass:
        try_im, energy_orig, energy = sess.run(
            output, {X_NOISE: orig_im, Y: label[0:1], LABEL: label})
    else:
        try_im, energy_orig, energy = sess.run(
            output, {X_NOISE: orig_im, Y: label[0:1]})

    orig_im = rescale_im(orig_im)
    try_im = rescale_im(try_im)
    actual_im = rescale_im(data)

    for i, (im, energy_i, t_im, energy, label_i, actual_im_i) in enumerate(
            zip(orig_im, energy_orig, try_im, energy, label, actual_im)):
        label_i = np.array(label_i)

        shape = im.shape[1:]
        new_im = np.zeros((shape[0], shape[1] * 3, *shape[2:]))
        size = shape[1]
        new_im[:, :size] = im
        new_im[:, size:2 * size] = t_im

        if FLAGS.cclass:
            label_i = np.where(label_i == 1)[0][0]
            if FLAGS.dataset == 'cifar10':
                log_image(new_im, logger, '{}_{:.4f}_now_{:.4f}_{}'.format(
                    i, energy_i[0], energy[0], cifar10_map[label_i]), step=i)
            else:
                log_image(
                    new_im,
                    logger,
                    '{}_{:.4f}_now_{:.4f}_{}'.format(
                        i,
                        energy_i[0],
                        energy[0],
                        label_i),
                    step=i)
        else:
            log_image(
                new_im,
                logger,
                '{}_{:.4f}_now_{:.4f}'.format(
                    i,
                    energy_i[0],
                    energy[0]),
                step=i)

    test_ims = list(try_im)
    real_ims = list(actual_im)

    for i in tqdm(range(50000 // FLAGS.batch_size + 1)):
        try:
            data_corrupt, data, label = dataloader_iterator.next()
        except BaseException:
            dataloader_iterator = iter(dataloader)
            data_corrupt, data, label = dataloader_iterator.next()

        data_corrupt, data, label = data_corrupt.numpy(), data.numpy(), label.numpy()

        if FLAGS.cclass:
            try_im, energy_orig, energy = sess.run(
                output, {X_NOISE: data_corrupt, Y: label[0:1], LABEL: label})
        else:
            try_im, energy_orig, energy = sess.run(
                output, {X_NOISE: data_corrupt, Y: label[0:1]})

        try_im = rescale_im(try_im)
        real_im = rescale_im(data)

        test_ims.extend(list(try_im))
        real_ims.extend(list(real_im))

    score, std = get_inception_score(test_ims)
    print("Inception score of {} with std of {}".format(score, std))


def main():

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    if hvd.rank() == 0:
        if not osp.exists(logdir):
            os.makedirs(logdir)
        logger = TensorBoardOutputFormat(logdir)
    else:
        logger = None

    LABEL = None
    print("Loading data...")
    if FLAGS.dataset == 'cifar10':
        dataset = Cifar10(augment=FLAGS.augment, rescale=FLAGS.rescale)
        test_dataset = Cifar10(train=False, rescale=FLAGS.rescale)
        channel_num = 3
        dim_input = 32 * 32 * 3

        X_NOISE = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
        X = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
        LABEL = tf.placeholder(shape=(None, 10), dtype=tf.float32)
        LABEL_POS = tf.placeholder(shape=(None, 10), dtype=tf.float32)

        if FLAGS.large_model:
            model = ResNet32Large(
                dim_input=dim_input,
                num_channels=channel_num,
                num_filters=128,
                dim_output=dim_output,
                train=True)
        elif FLAGS.larger_model:
            model = ResNet32Larger(
                dim_input=dim_input,
                num_channels=channel_num,
                num_filters=128,
                dim_output=dim_output)
        elif FLAGS.wider_model:
            model = ResNet32Wider(
                dim_input=dim_input,
                num_channels=channel_num,
                num_filters=192,
                dim_output=dim_output)
        else:
            model = ResNet32(
                dim_input=dim_input,
                num_channels=channel_num,
                num_filters=128,
                dim_output=dim_output)

    elif FLAGS.dataset == 'imagenet':
        dataset = Imagenet()
        channel_num = 3
        dim_input = 32 * 32 * 3
        X_NOISE = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
        X = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
        LABEL = tf.placeholder(shape=(None, 1000), dtype=tf.float32)
        LABEL_POS = tf.placeholder(shape=(None, 1000), dtype=tf.float32)

        model = ResNet32Wider(
            dim_input=dim_input,
            num_channels=channel_num,
            num_filters=256,
            dim_output=dim_output)

    elif FLAGS.dataset == 'mnist':
        dataset = Mnist()
        test_dataset = dataset
        channel_num = 1
        dim_input = 28 * 28 * 1
        X_NOISE = tf.placeholder(shape=(None, 28, 28), dtype=tf.float32)
        X = tf.placeholder(shape=(None, 28, 28), dtype=tf.float32)
        LABEL = tf.placeholder(shape=(None, 10), dtype=tf.float32)
        LABEL_POS = tf.placeholder(shape=(None, 10), dtype=tf.float32)

        model = MnistNet(
            dim_input=dim_input,
            num_channels=channel_num,
            num_filters=FLAGS.num_filters,
            dim_output=dim_output)

    elif FLAGS.dataset == 'dsprites':
        dataset = DSprites(
            cond_shape=FLAGS.cond_shape,
            cond_size=FLAGS.cond_size,
            cond_pos=FLAGS.cond_pos,
            cond_rot=FLAGS.cond_rot)
        test_dataset = dataset
        channel_num = 1
        dim_input = 64 * 64 * 1
        dim_output = 1

        X_NOISE = tf.placeholder(shape=(None, 64, 64), dtype=tf.float32)
        X = tf.placeholder(shape=(None, 64, 64), dtype=tf.float32)

        if FLAGS.dpos_only:
            LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        elif FLAGS.dsize_only:
            LABEL = tf.placeholder(shape=(None, 1), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        elif FLAGS.drot_only:
            LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        elif FLAGS.cond_size:
            LABEL = tf.placeholder(shape=(None, 1), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        elif FLAGS.cond_shape:
            LABEL = tf.placeholder(shape=(None, 3), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 3), dtype=tf.float32)
        elif FLAGS.cond_pos:
            LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        elif FLAGS.cond_rot:
            LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        else:
            LABEL = tf.placeholder(shape=(None, 3), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 3), dtype=tf.float32)

        model = DspritesNet(
            dim_input=dim_input,
            num_channels=channel_num,
            num_filters=FLAGS.num_filters,
            dim_output=dim_output,
            cond_size=FLAGS.cond_size,
            cond_shape=FLAGS.cond_shape,
            cond_pos=FLAGS.cond_pos,
            cond_rot=FLAGS.cond_rot)

    print("Done loading...")
    data_loader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.data_workers,
        drop_last=True,
        shuffle=True)
    data_loader_test = DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.data_workers,
        drop_last=True,
        shuffle=True)

    batch_size = FLAGS.batch_size

    weights = [model.construct_weights('context_0')]

    Y = tf.placeholder(shape=(None), dtype=tf.int32)

    # Varibles to run in training
    X_SPLIT = tf.split(X, FLAGS.num_gpus)
    X_NOISE_SPLIT = tf.split(X_NOISE, FLAGS.num_gpus)
    LABEL_SPLIT = tf.split(LABEL, FLAGS.num_gpus)
    LABEL_POS_SPLIT = tf.split(LABEL_POS, FLAGS.num_gpus)
    LABEL_SPLIT_INIT = list(LABEL_SPLIT)
    tower_grads = []
    tower_gen_grads = []
    x_mod_list = []

    optimizer = AdamOptimizer(FLAGS.lr, beta1=0.0, beta2=0.999)
    optimizer = hvd.DistributedOptimizer(optimizer)

    for j in range(FLAGS.num_gpus):

        if FLAGS.model_cclass:
            ind_batch_size = FLAGS.batch_size // FLAGS.num_gpus
            label_tensor = tf.Variable(
                tf.convert_to_tensor(
                    np.reshape(
                        np.tile(np.eye(10), (FLAGS.batch_size, 1, 1)),
                        (FLAGS.batch_size * 10, 10)),
                    dtype=tf.float32),
                trainable=False,
                dtype=tf.float32)
            x_split = tf.tile(
                tf.reshape(
                    X_SPLIT[j], (ind_batch_size, 1, 32, 32, 3)), (1, 10, 1, 1, 1))
            x_split = tf.reshape(x_split, (ind_batch_size * 10, 32, 32, 3))
            energy_pos = model.forward(
                x_split,
                weights[0],
                label=label_tensor,
                stop_at_grad=False)

            energy_pos_full = tf.reshape(energy_pos, (ind_batch_size, 10))
            energy_partition_est = tf.reduce_logsumexp(
                energy_pos_full, axis=1, keepdims=True)
            uniform = tf.random_uniform(tf.shape(energy_pos_full))
            label_tensor = tf.argmax(-energy_pos_full -
                                     tf.log(-tf.log(uniform)) - energy_partition_est, axis=1)
            label = tf.one_hot(label_tensor, 10, dtype=tf.float32)
            label = tf.Print(label, [label_tensor, energy_pos_full])
            LABEL_SPLIT[j] = label
            energy_pos = tf.concat(energy_pos, axis=0)
        else:
            energy_pos = [
                model.forward(
                    X_SPLIT[j],
                    weights[0],
                    label=LABEL_POS_SPLIT[j],
                    stop_at_grad=False)]
            energy_pos = tf.concat(energy_pos, axis=0)

        print("Building graph...")
        x_mod = x_orig = X_NOISE_SPLIT[j]

        x_grads = []

        energy_negs = []
        loss_energys = []

        energy_negs.extend([model.forward(tf.stop_gradient(
            x_mod), weights[0], label=LABEL_SPLIT[j], stop_at_grad=False, reuse=True)])
        eps_begin = tf.zeros(1)

        for i in range(FLAGS.num_steps):
            x_mod = x_mod + tf.random_normal(tf.shape(x_mod),
                                             mean=0.0,
                                             stddev=0.005 * FLAGS.rescale * FLAGS.noise_scale)

            energy_noise = energy_start = tf.concat(
                [model.forward(
                        x_mod,
                        weights[0],
                        label=LABEL_SPLIT[j],
                        reuse=True,
                        stop_at_grad=True,
                        stop_batch=True)],
                axis=0)

            x_grad, label_grad = tf.gradients(
                FLAGS.temperature * energy_noise, [x_mod, LABEL_SPLIT[j]])
            energy_noise_old = energy_noise

            x_grads.append(x_grad)
            lr = FLAGS.step_lr

            if FLAGS.proj_norm != 0.0:
                if FLAGS.proj_norm_type == 'l2':
                    x_grad = tf.clip_by_norm(x_grad, FLAGS.proj_norm)
                elif FLAGS.proj_norm_type == 'li':
                    x_grad = tf.clip_by_value(
                        x_grad, -FLAGS.proj_norm, FLAGS.proj_norm)
                else:
                    print("Other types of projection are not supported!!!")
                    assert False

                if FLAGS.stop_proj_norm:
                    x_grad = tf.stop_gradient(x_grad)

            def energy(x):
                return FLAGS.temperature * \
                    model.forward(x, weights[0], label=LABEL_SPLIT[j], reuse=True)

            # Clip gradient norm for now
            if FLAGS.hmc:
                # Step size should be tuned to get around 65% acceptance
                x_last = hmc(x_mod, 0.0045, 10, energy)
            else:
                x_last = x_mod - (lr) * x_grad

            x_mod = x_last
            x_mod = tf.clip_by_value(x_mod, 0, FLAGS.rescale)

            print("Building loop {} ...".format(i))

            energy_negs.append(
                model.forward(
                    tf.stop_gradient(x_mod),
                    weights[0],
                    label=LABEL_SPLIT[j],
                    stop_at_grad=False,
                    reuse=True))

        test_x_mod = x_mod

        temp = FLAGS.temperature

        energy_neg = energy_negs[-1]

        loss_energy = [tf.clip_by_value(
            temp * model.forward(x_mod, weights[0], reuse=True, label=LABEL, stop_grad=True), -1e5, 1e5)]
        loss_energy = tf.concat(loss_energy, axis=0)
        x_off = tf.reduce_mean(
            tf.abs(x_mod[:tf.shape(X_SPLIT[j])[0]] - X_SPLIT[j]))
        loss_energy = model.forward(
            x_mod,
            weights[0],
            reuse=True,
            label=LABEL,
            stop_grad=True)

        print("Finished processing loop construction ...")

        target_vars = {}

        if FLAGS.cclass or FLAGS.model_cclass:
            label_sum = tf.reduce_sum(LABEL_SPLIT[0], axis=0)
            label_prob = label_sum / tf.reduce_sum(label_sum)
            label_ent = -tf.reduce_sum(label_prob *
                                       tf.math.log(label_prob + 1e-7))
        else:
            label_ent = tf.zeros(1)

        target_vars['label_ent'] = label_ent

        if FLAGS.train:

            if FLAGS.objective == 'logsumexp':
                pos_term = temp * energy_pos
                energy_neg_reduced = (energy_neg - tf.reduce_min(energy_neg))
                coeff = tf.stop_gradient(tf.exp(-temp * energy_neg_reduced))
                norm_constant = tf.stop_gradient(tf.reduce_sum(coeff)) + 1e-4
                pos_loss = tf.reduce_mean(temp * energy_pos)
                neg_loss = coeff * (-1 * temp * energy_neg) / norm_constant
                loss_ml = FLAGS.ml_coeff * (pos_loss + tf.reduce_sum(neg_loss))
            elif FLAGS.objective == 'cd':
                pos_loss = tf.reduce_mean(temp * energy_pos)
                neg_loss = -tf.reduce_mean(temp * energy_neg)
                loss_ml = FLAGS.ml_coeff * (pos_loss + tf.reduce_sum(neg_loss))
            elif FLAGS.objective == 'softplus':
                loss_ml = FLAGS.ml_coeff * \
                    tf.nn.softplus(temp * (energy_pos - energy_neg))

            loss_total = tf.reduce_mean(loss_ml)

            if not FLAGS.zero_kl:
                loss_total = loss_total + tf.reduce_mean(loss_energy)

            loss_total = loss_total + \
                (tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(tf.square((energy_neg))))

            print("Started gradient computation...")
            gvs = optimizer.compute_gradients(loss_total)
            gvs = [(k, v) for (k, v) in gvs if k is not None]

            print("Applying gradients...")

            tower_grads.append(gvs)

            print("Finished applying gradients.")

            target_vars['loss_ml'] = loss_ml
            target_vars['total_loss'] = loss_total
            target_vars['loss_energy'] = loss_energy
            target_vars['weights'] = weights
            target_vars['gvs'] = gvs

        target_vars['X'] = X
        target_vars['Y'] = Y
        target_vars['LABEL'] = LABEL
        target_vars['LABEL_POS'] = LABEL_POS
        target_vars['X_NOISE'] = X_NOISE
        target_vars['energy_pos'] = energy_pos
        target_vars['energy_start'] = energy_negs[0]

        if len(x_grads) > 1:
            target_vars['x_grad'] = x_grads[-1]
            target_vars['x_grad_first'] = x_grads[0]
        else:
            target_vars['x_grad'] = tf.zeros(1)
            target_vars['x_grad_first'] = tf.zeros(1)

        target_vars['x_mod'] = x_mod
        target_vars['x_off'] = x_off
        target_vars['temp'] = temp
        target_vars['energy_neg'] = energy_neg
        target_vars['test_x_mod'] = test_x_mod
        target_vars['eps_begin'] = eps_begin

    if FLAGS.train:
        grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(grads)
        target_vars['train_op'] = train_op

    config = tf.ConfigProto()

    if hvd.size() > 1:
        config.gpu_options.visible_device_list = str(hvd.local_rank())

    sess = tf.Session(config=config)

    saver = loader = tf.train.Saver(
        max_to_keep=10, keep_checkpoint_every_n_hours=6)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Model has a total of {} parameters".format(total_parameters))

    sess.run(tf.global_variables_initializer())

    resume_itr = 0

    if (FLAGS.resume_iter != -1 or not FLAGS.train) and hvd.rank() == 0:
        model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
        resume_itr = FLAGS.resume_iter
        saver.restore(sess, model_file)

    sess.run(hvd.broadcast_global_variables(0))
    print("Initializing variables...")

    print("Start broadcast")
    print("End broadcast")

    if FLAGS.train:
        train(target_vars, saver, sess,
              logger, data_loader, resume_itr,
              logdir)

    test(target_vars, saver, sess, logger, data_loader)


if __name__ == "__main__":
    main()
