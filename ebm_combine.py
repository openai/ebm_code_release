import tensorflow as tf
import math
from tqdm import tqdm
from hmc import hmc
from tensorflow.python.platform import flags
from torch.utils.data import DataLoader, Dataset
from models import DspritesNet
from utils import optimistic_restore, ReplayBuffer
import os.path as osp
import numpy as np
from rl_algs.logger import TensorBoardOutputFormat
from scipy.misc import imsave
import os
from custom_adam import AdamOptimizer

flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_integer('data_workers', 4, 'Number of workers to do things')
flags.DEFINE_string('logdir', 'cachedir', 'directory for logging')
flags.DEFINE_string('savedir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omniglot.')
flags.DEFINE_float('step_lr', 500, 'size of gradient descent size')
flags.DEFINE_string('dsprites_path', '/root/data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', 'path to dsprites characters')
flags.DEFINE_bool('cclass', True, 'not cclass')
flags.DEFINE_bool('proj_cclass', False, 'use for backwards compatibility reasons')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_bias', True, 'Whether to use bias in convolution')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_bool('plot_curve', False, 'Generate a curve of results')
flags.DEFINE_integer('num_steps', 20, 'number of steps to optimize the label')
flags.DEFINE_string('task', 'conceptcombine', 'conceptcombine, labeldiscover, gentest, genbaseline, etc.')
flags.DEFINE_bool('joint_shape', False, 'whether to use pos_size or pos_shape')
flags.DEFINE_bool('joint_rot', False, 'whether to use pos_size or pos_shape')

# Conditions on which models to use
flags.DEFINE_bool('cond_pos', True, 'whether to condition on position')
flags.DEFINE_bool('cond_rot', True, 'whether to condition on rotation')
flags.DEFINE_bool('cond_shape', True, 'whether to condition on shape')
flags.DEFINE_bool('cond_scale', True, 'whether to condition on scale')

flags.DEFINE_string('exp_size', 'dsprites_2018_cond_size', 'name of experiments')
flags.DEFINE_string('exp_shape', 'dsprites_2018_cond_shape', 'name of experiments')
flags.DEFINE_string('exp_pos', 'dsprites_2018_cond_pos_cert', 'name of experiments')
flags.DEFINE_string('exp_rot', 'dsprites_cond_rot_119_00', 'name of experiments')
flags.DEFINE_integer('resume_size', 169000, 'First iteration to resume')
flags.DEFINE_integer('resume_shape', 477000, 'Second iteration to resume')
flags.DEFINE_integer('resume_pos', 8000, 'Second iteration to resume')
flags.DEFINE_integer('resume_rot', 690000, 'Second iteration to resume')
flags.DEFINE_integer('break_steps', 300, 'steps to break')

# Whether to train for gentest
flags.DEFINE_bool('train', False, 'whether to train on generalization into multiple different predictions')

FLAGS = flags.FLAGS

class DSpritesGen(Dataset):
    def __init__(self, data, latents, frac=0.0):

        l = latents

        if FLAGS.joint_shape:
            mask_size = (l[:, 3] == 30 * np.pi / 39) & (l[:, 4] == 16/31) & (l[:, 5] == 16/31) & (l[:, 2] == 0.5)
        elif FLAGS.joint_rot:
            mask_size = (l[:, 1] == 1) & (l[:, 4] == 16/31) & (l[:, 5] == 16/31) & (l[:, 2] == 0.5)
        else:
            mask_size = (l[:, 3] == 30 * np.pi / 39) & (l[:, 4] == 16/31) & (l[:, 5] == 16/31) & (l[:, 1] == 1)

        mask_pos = (l[:, 1] == 1) & (l[:, 3] == 30 * np.pi / 39) & (l[:, 2] == 0.5)

        data_pos = data[mask_pos]
        l_pos = l[mask_pos]

        data_size = data[mask_size]
        l_size = l[mask_size]

        n = data_pos.shape[0] // data_size.shape[0]

        data_pos = np.tile(data_pos, (n, 1, 1))
        l_pos = np.tile(l_pos, (n, 1))

        self.data = np.concatenate((data_pos, data_size), axis=0)
        self.label = np.concatenate((l_pos, l_size), axis=0)

        mask_neg = (~(mask_size & mask_pos)) & ((l[:, 1] == 1) & (l[:, 3] == 30 * np.pi / 39))
        data_add = data[mask_neg]
        l_add = l[mask_neg]

        perm_idx = np.random.permutation(data_add.shape[0])
        select_idx = perm_idx[:int(frac*perm_idx.shape[0])]
        data_add = data_add[select_idx]
        l_add = l_add[select_idx]

        self.data = np.concatenate((self.data, data_add), axis=0)
        self.label = np.concatenate((self.label, l_add), axis=0)

        self.identity = np.eye(3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        im = self.data[index]
        im_corrupt = 0.5 + 0.5 * np.random.randn(64, 64)

        if FLAGS.joint_shape:
            label_size = np.eye(3)[self.label[index, 1].astype(np.int32) - 1]
        elif FLAGS.joint_rot:
            label_size = np.array([np.cos(self.label[index, 3]), np.sin(self.label[index, 3])])
        else:
            label_size = self.label[index, 2:3]

        label_pos = self.label[index, 4:]

        return (im_corrupt, im, label_size, label_pos)


def labeldiscover(sess, kvs, data, latents, save_exp_dir):
    LABEL_SIZE = kvs['LABEL_SIZE']
    model_size = kvs['model_size']
    weight_size = kvs['weight_size']
    x_mod = kvs['X_NOISE']

    label_output = LABEL_SIZE
    for i in range(FLAGS.num_steps):
        label_output = label_output + tf.random_normal(tf.shape(label_output), mean=0.0, stddev=0.03)
        e_noise = model_size.forward(x_mod, weight_size, label=label_output)
        label_grad = tf.gradients(e_noise, [label_output])[0]
        # label_grad = tf.Print(label_grad, [label_grad])
        label_output = label_output - 1.0 * label_grad
        label_output = tf.clip_by_value(label_output, 0.5, 1.0)

    diffs = []
    for i in range(30):
        s = i*FLAGS.batch_size
        d = (i+1)*FLAGS.batch_size
        data_i = data[s:d]
        latent_i = latents[s:d]
        latent_init = np.random.uniform(0.5, 1, (FLAGS.batch_size, 1))

        feed_dict = {x_mod: data_i, LABEL_SIZE:latent_init}
        size_pred = sess.run([label_output], feed_dict)[0]
        size_gt = latent_i[:, 2:3]

        diffs.append(np.abs(size_pred - size_gt).mean())

    print(np.array(diffs).mean())


def genbaseline(sess, kvs, data, latents, save_exp_dir, frac=0.0):
    # tf.reset_default_graph()

    if FLAGS.joint_shape:
        model_baseline = DspritesNetGen(num_filters=FLAGS.num_filters, label_size=5)
        LABEL = tf.placeholder(shape=(None, 5), dtype=tf.float32)
    else:
        model_baseline = DspritesNetGen(num_filters=FLAGS.num_filters, label_size=3)
        LABEL = tf.placeholder(shape=(None, 3), dtype=tf.float32)

    weights_baseline = model_baseline.construct_weights('context_baseline_{}'.format(frac))

    X_feed = tf.placeholder(shape=(None, 2*FLAGS.num_filters), dtype=tf.float32)
    X_label = tf.placeholder(shape=(None, 64, 64), dtype=tf.float32)

    X_out = model_baseline.forward(X_feed, weights_baseline, label=LABEL)
    loss_sq = tf.reduce_mean(tf.square(X_out - X_label))

    optimizer = AdamOptimizer(1e-3)
    gvs = optimizer.compute_gradients(loss_sq)
    gvs = [(k, v) for (k, v) in gvs if k is not None]
    train_op = optimizer.apply_gradients(gvs)

    dataloader = DataLoader(DSpritesGen(data, latents, frac=frac), batch_size=FLAGS.batch_size, num_workers=6, drop_last=True, shuffle=True)

    datafull = data

    itr = 0
    saver = tf.train.Saver()

    vs = optimizer.variables()
    sess.run(tf.global_variables_initializer())

    if FLAGS.train:
        for _ in range(5):
            for data_corrupt, data, label_size, label_pos in tqdm(dataloader):

                data_corrupt = data_corrupt.numpy()
                label_size, label_pos = label_size.numpy(), label_pos.numpy()

                data_corrupt = np.random.randn(data_corrupt.shape[0], 2*FLAGS.num_filters)
                label_comb = np.concatenate([label_size, label_pos], axis=1)

                feed_dict = {X_feed: data_corrupt, X_label: data, LABEL: label_comb}

                output = [loss_sq, train_op]

                loss, _ = sess.run(output, feed_dict=feed_dict)

                itr += 1

        saver.save(sess, osp.join(save_exp_dir, 'model_genbaseline'))

    saver.restore(sess, osp.join(save_exp_dir, 'model_genbaseline'))

    l = latents

    if FLAGS.joint_shape:
        mask_gen = (l[:, 3] == 30 * np.pi / 39) * (l[:, 2] == 0.5)
    else:
        mask_gen = (l[:, 3] == 30 * np.pi / 39) * (l[:, 1] == 1) & (~((l[:, 2] == 0.5) | ((l[:, 4] == 16/31) & (l[:, 5] == 16/31))))

    data_gen = datafull[mask_gen]
    latents_gen = latents[mask_gen]
    losses = []

    for dat, latent in zip(np.array_split(data_gen, 10), np.array_split(latents_gen, 10)):
        data_init = np.random.randn(dat.shape[0], 2*FLAGS.num_filters)

        if FLAGS.joint_shape:
            latent_size = np.eye(3)[latent[:, 1].astype(np.int32) - 1]
            latent_pos = latent[:, 4:6]
            latent = np.concatenate([latent_size, latent_pos], axis=1)
            feed_dict = {X_feed: data_init, LABEL: latent, X_label: dat}
        else:
            feed_dict = {X_feed: data_init, LABEL: latent[:, [2,4,5]], X_label: dat}
        loss = sess.run([loss_sq], feed_dict=feed_dict)[0]
        # print(loss)
        losses.append(loss)

    print("Overall MSE for generalization of {} for fraction of {}".format(np.mean(losses), frac))


    data_try = data_gen[:10]
    data_init = np.random.randn(10, 2*FLAGS.num_filters)

    if FLAGS.joint_shape:
        latent_scale = np.eye(3)[latent[:10, 1].astype(np.int32) - 1]
        latent_pos = latents_gen[:10, 4:]
    else:
        latent_scale = latents_gen[:10, 2:3]
        latent_pos = latents_gen[:10, 4:]

    latent_tot = np.concatenate([latent_scale, latent_pos], axis=1)

    feed_dict = {X_feed: data_init, LABEL: latent_tot}
    x_output = sess.run([X_out], feed_dict=feed_dict)[0]
    x_output = np.clip(x_output, 0, 1)

    im_name = "size_scale_combine_genbaseline.png"

    x_output_wrap = np.ones((10, 66, 66))
    data_try_wrap = np.ones((10, 66, 66))

    x_output_wrap[:, 1:-1, 1:-1] = x_output
    data_try_wrap[:, 1:-1, 1:-1] = data_try

    im_output = np.concatenate([x_output_wrap, data_try_wrap], axis=2).reshape(-1, 66*2)
    impath = osp.join(save_exp_dir, im_name)
    imsave(impath, im_output)
    print("Successfully saved images at {}".format(impath))

    return np.mean(losses)


def gentest(sess, kvs, data, latents, save_exp_dir):
    X_NOISE = kvs['X_NOISE']
    LABEL_SIZE = kvs['LABEL_SIZE']
    LABEL_SHAPE = kvs['LABEL_SHAPE']
    LABEL_POS = kvs['LABEL_POS']
    LABEL_ROT = kvs['LABEL_ROT']
    model_size = kvs['model_size']
    model_shape = kvs['model_shape']
    model_pos = kvs['model_pos']
    model_rot = kvs['model_rot']
    weight_size = kvs['weight_size']
    weight_shape = kvs['weight_shape']
    weight_pos = kvs['weight_pos']
    weight_rot = kvs['weight_rot']
    X = tf.placeholder(shape=(None, 64, 64), dtype=tf.float32)

    datafull = data
    # Test combination of generalization where we use slices of both training
    x_final = X_NOISE
    x_mod_size = X_NOISE
    x_mod_pos = X_NOISE

    for i in range(FLAGS.num_steps):

        # use cond_pos

        energies = []
        x_mod_pos = x_mod_pos + tf.random_normal(tf.shape(x_mod_pos), mean=0.0, stddev=0.005)
        e_noise = model_pos.forward(x_final, weight_pos, label=LABEL_POS)

        # energies.append(e_noise)
        x_grad = tf.gradients(e_noise, [x_final])[0]
        x_mod_pos = x_mod_pos + tf.random_normal(tf.shape(x_mod_pos), mean=0.0, stddev=0.005)
        x_mod_pos = x_mod_pos - FLAGS.step_lr * x_grad
        x_mod_pos = tf.clip_by_value(x_mod_pos, 0, 1)

        if FLAGS.joint_shape:
            # use cond_shape
            e_noise = model_shape.forward(x_mod_pos, weight_shape, label=LABEL_SHAPE)
        elif FLAGS.joint_rot:
            e_noise = model_rot.forward(x_mod_pos, weight_rot, label=LABEL_ROT)
        else:
            # use cond_size
            e_noise = model_size.forward(x_mod_pos, weight_size, label=LABEL_SIZE)

        # energies.append(e_noise)
        # energy_stack = tf.concat(energies, axis=1)
        # energy_stack = tf.reduce_logsumexp(-1*energy_stack, axis=1)
        # energy_stack = tf.reduce_sum(energy_stack, axis=1)

        x_grad = tf.gradients(e_noise, [x_mod_pos])[0]
        x_mod_pos = x_mod_pos - FLAGS.step_lr * x_grad
        x_mod_pos = tf.clip_by_value(x_mod_pos, 0, 1)

        # for x_mod_size
        # use cond_size
        # e_noise = model_size.forward(x_mod_size, weight_size, label=LABEL_SIZE)
        # x_grad = tf.gradients(e_noise, [x_mod_size])[0]
        # x_mod_size = x_mod_size + tf.random_normal(tf.shape(x_mod_size), mean=0.0, stddev=0.005)
        # x_mod_size = x_mod_size - FLAGS.step_lr * x_grad
        # x_mod_size = tf.clip_by_value(x_mod_size, 0, 1)

        # # use cond_pos
        # e_noise = model_pos.forward(x_mod_size, weight_pos, label=LABEL_POS)
        # x_grad = tf.gradients(e_noise, [x_mod_size])[0]
        # x_mod_size = x_mod_size + tf.random_normal(tf.shape(x_mod_size), mean=0.0, stddev=0.005)
        # x_mod_size = x_mod_size - FLAGS.step_lr * tf.stop_gradient(x_grad)
        # x_mod_size = tf.clip_by_value(x_mod_size, 0, 1)

    x_mod = x_mod_pos
    x_final = x_mod


    if FLAGS.joint_shape:
        loss_kl = model_shape.forward(x_final, weight_shape, reuse=True, label=LABEL_SHAPE, stop_grad=True) + \
                  model_pos.forward(x_final, weight_pos, reuse=True, label=LABEL_POS, stop_grad=True)

        energy_pos = model_shape.forward(X, weight_shape, reuse=True, label=LABEL_SHAPE) + \
                      model_pos.forward(X, weight_pos, reuse=True, label=LABEL_POS)

        energy_neg = model_shape.forward(tf.stop_gradient(x_mod), weight_shape, reuse=True, label=LABEL_SHAPE) + \
                      model_pos.forward(tf.stop_gradient(x_mod), weight_pos, reuse=True, label=LABEL_POS)
    elif FLAGS.joint_rot:
        loss_kl = model_rot.forward(x_final, weight_rot, reuse=True, label=LABEL_ROT, stop_grad=True) + \
                  model_pos.forward(x_final, weight_pos, reuse=True, label=LABEL_POS, stop_grad=True)

        energy_pos = model_rot.forward(X, weight_rot, reuse=True, label=LABEL_ROT) + \
                      model_pos.forward(X, weight_pos, reuse=True, label=LABEL_POS)

        energy_neg = model_rot.forward(tf.stop_gradient(x_mod), weight_rot, reuse=True, label=LABEL_ROT) + \
                      model_pos.forward(tf.stop_gradient(x_mod), weight_pos, reuse=True, label=LABEL_POS)
    else:
        loss_kl = model_size.forward(x_final, weight_size, reuse=True, label=LABEL_SIZE, stop_grad=True) + \
                    model_pos.forward(x_final, weight_pos, reuse=True, label=LABEL_POS, stop_grad=True)

        energy_pos = model_size.forward(X, weight_size, reuse=True, label=LABEL_SIZE) + \
                      model_pos.forward(X, weight_pos, reuse=True, label=LABEL_POS)

        energy_neg = model_size.forward(tf.stop_gradient(x_mod), weight_size, reuse=True, label=LABEL_SIZE) + \
                      model_pos.forward(tf.stop_gradient(x_mod), weight_pos, reuse=True, label=LABEL_POS)

    energy_neg_reduced = (energy_neg - tf.reduce_min(energy_neg))
    coeff = tf.stop_gradient(tf.exp(-energy_neg_reduced))
    norm_constant = tf.stop_gradient(tf.reduce_sum(coeff)) + 1e-4
    neg_loss = coeff * (-1*energy_neg) / norm_constant

    loss_ml = tf.reduce_mean(energy_pos) - tf.reduce_mean(energy_neg)
    loss_total = loss_ml + tf.reduce_mean(loss_kl) + 1 * (tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(tf.square(energy_neg)))

    optimizer = AdamOptimizer(1e-3, beta1=0.0, beta2=0.999)
    gvs = optimizer.compute_gradients(loss_total)
    gvs = [(k, v) for (k, v) in gvs if k is not None]
    train_op = optimizer.apply_gradients(gvs)

    vs = optimizer.variables()
    sess.run(tf.variables_initializer(vs))

    dataloader = DataLoader(DSpritesGen(data, latents), batch_size=FLAGS.batch_size, num_workers=6, drop_last=True, shuffle=True)

    x_off = tf.reduce_mean(tf.square(x_mod - X))

    itr = 0
    saver = tf.train.Saver()
    x_mod = None


    if FLAGS.train:
        replay_buffer = ReplayBuffer(10000)
        for _ in range(1):


            for data_corrupt, data, label_size, label_pos in tqdm(dataloader):
                data_corrupt = data_corrupt.numpy()[:, :, :]
                data = data.numpy()[:, :, :]

                if x_mod is not None:
                    replay_buffer.add(x_mod)
                    replay_batch = replay_buffer.sample(FLAGS.batch_size)
                    replay_mask = (np.random.uniform(0, 1, (FLAGS.batch_size)) > 0.95)
                    data_corrupt[replay_mask] = replay_batch[replay_mask]

                if FLAGS.joint_shape:
                    feed_dict = {X_NOISE: data_corrupt, X: data, LABEL_SHAPE: label_size, LABEL_POS: label_pos}
                elif FLAGS.joint_rot:
                    feed_dict = {X_NOISE: data_corrupt, X: data, LABEL_ROT: label_size, LABEL_POS: label_pos}
                else:
                    feed_dict = {X_NOISE: data_corrupt, X: data, LABEL_SIZE: label_size, LABEL_POS: label_pos}

                _, off_value, e_pos, e_neg, x_mod = sess.run([train_op, x_off, energy_pos, energy_neg, x_final], feed_dict=feed_dict)
                itr += 1

                if itr % 10 == 0:
                    print("x_off of {}, e_pos of {}, e_neg of {} itr of {}".format(off_value, e_pos.mean(), e_neg.mean(), itr))

                if itr == FLAGS.break_steps:
                    break


        saver.save(sess, osp.join(save_exp_dir, 'model_gentest'))

    saver.restore(sess, osp.join(save_exp_dir, 'model_gentest'))

    l = latents

    if FLAGS.joint_shape:
        mask_gen = (l[:, 3] == 30 * np.pi / 39) * (l[:, 2] == 0.5)
    elif FLAGS.joint_rot:
        mask_gen = (l[:, 1] == 1) * (l[:, 2] == 0.5)
    else:
        mask_gen = (l[:, 3] == 30 * np.pi / 39) * (l[:, 1] == 1) & (~((l[:, 2] == 0.5) | ((l[:, 4] == 16/31) & (l[:, 5] == 16/31))))

    data_gen = datafull[mask_gen]
    latents_gen = latents[mask_gen]

    losses = []

    for dat, latent in zip(np.array_split(data_gen, 120), np.array_split(latents_gen, 120)):
        x = 0.5 + np.random.randn(*dat.shape)

        if FLAGS.joint_shape:
            feed_dict = {LABEL_SHAPE: np.eye(3)[latent[:, 1].astype(np.int32) - 1], LABEL_POS: latent[:, 4:], X_NOISE: x, X: dat}
        elif FLAGS.joint_rot:
            feed_dict = {LABEL_ROT: np.concatenate([np.cos(latent[:, 3:4]), np.sin(latent[:, 3:4])], axis=1), LABEL_POS: latent[:, 4:], X_NOISE: x, X: dat}
        else:
            feed_dict = {LABEL_SIZE: latent[:, 2:3], LABEL_POS: latent[:, 4:], X_NOISE: x, X: dat}

        for i in range(2):
            x = sess.run([x_final], feed_dict=feed_dict)[0]
            feed_dict[X_NOISE] = x

        loss = sess.run([x_off], feed_dict=feed_dict)[0]
        losses.append(loss)

    print("Mean MSE loss of {} ".format(np.mean(losses)))

    data_try = data_gen[:10]
    data_init = 0.5 + 0.5 * np.random.randn(10, 64, 64)
    latent_scale = latents_gen[:10, 2:3]
    latent_pos = latents_gen[:10, 4:]

    if FLAGS.joint_shape:
        feed_dict = {X_NOISE: data_init, LABEL_SHAPE: np.eye(3)[latent[:10, 1].astype(np.int32)-1], LABEL_POS: latent_pos}
    elif FLAGS.joint_rot:
        feed_dict = {LABEL_ROT: np.concatenate([np.cos(latent[:10, 3:4]), np.sin(latent[:10, 3:4])], axis=1), LABEL_POS: latent[:10, 4:], X_NOISE: data_init}
    else:
        feed_dict = {X_NOISE: data_init, LABEL_SIZE: latent_scale, LABEL_POS: latent_pos}

    x_output = sess.run([x_final], feed_dict=feed_dict)[0]

    if FLAGS.joint_shape:
        im_name = "size_shape_combine_gentest.png"
    else:
        im_name = "size_scale_combine_gentest.png"

    x_output_wrap = np.ones((10, 66, 66))
    data_try_wrap = np.ones((10, 66, 66))

    x_output_wrap[:, 1:-1, 1:-1] = x_output
    data_try_wrap[:, 1:-1, 1:-1] = data_try

    im_output = np.concatenate([x_output_wrap, data_try_wrap], axis=2).reshape(-1, 66*2)
    impath = osp.join(save_exp_dir, im_name)
    imsave(impath, im_output)
    print("Successfully saved images at {}".format(impath))



def conceptcombine(sess, kvs, data, latents, save_exp_dir):
    X_NOISE = kvs['X_NOISE']
    LABEL_SIZE = kvs['LABEL_SIZE']
    LABEL_SHAPE = kvs['LABEL_SHAPE']
    LABEL_POS = kvs['LABEL_POS']
    LABEL_ROT = kvs['LABEL_ROT']
    model_size = kvs['model_size']
    model_shape = kvs['model_shape']
    model_pos = kvs['model_pos']
    model_rot = kvs['model_rot']
    weight_size = kvs['weight_size']
    weight_shape = kvs['weight_shape']
    weight_pos = kvs['weight_pos']
    weight_rot = kvs['weight_rot']

    x_mod = X_NOISE
    for i in range(FLAGS.num_steps):

        if FLAGS.cond_scale:
            e_noise = model_size.forward(x_mod, weight_size, label=LABEL_SIZE)
            x_grad = tf.gradients(e_noise, [x_mod])[0]
            x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.005)
            x_mod = x_mod - FLAGS.step_lr * x_grad
            x_mod = tf.clip_by_value(x_mod, 0, 1)

        if FLAGS.cond_shape:
            e_noise = model_shape.forward(x_mod, weight_shape, label=LABEL_SHAPE)
            x_grad = tf.gradients(e_noise, [x_mod])[0]
            x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.005)
            x_mod = x_mod - FLAGS.step_lr * x_grad
            x_mod = tf.clip_by_value(x_mod, 0, 1)

        if FLAGS.cond_pos:
            e_noise = model_pos.forward(x_mod, weight_pos, label=LABEL_POS)
            x_grad = tf.gradients(e_noise, [x_mod])[0]
            x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.005)
            x_mod = x_mod - FLAGS.step_lr * x_grad
            x_mod = tf.clip_by_value(x_mod, 0, 1)

        if FLAGS.cond_rot:
            e_noise = model_rot.forward(x_mod, weight_rot, label=LABEL_ROT)
            x_grad = tf.gradients(e_noise, [x_mod])[0]
            x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.005)
            x_mod = x_mod - FLAGS.step_lr * x_grad
            x_mod = tf.clip_by_value(x_mod, 0, 1)

        print("Finished constructing loop {}".format(i))

    x_final = x_mod

    data_try = data[:10]
    data_init = 0.5 + 0.5 * np.random.randn(10, 64, 64)
    label_scale = latents[:10, 2:3]
    label_shape = np.eye(3)[(latents[:10, 1]-1).astype(np.uint8)]
    label_rot = latents[:10, 3:4]
    label_rot = np.concatenate([np.cos(label_rot), np.sin(label_rot)], axis=1)
    label_pos = latents[:10, 4:]

    feed_dict = {X_NOISE: data_init, LABEL_SIZE: label_scale, LABEL_SHAPE: label_shape, LABEL_POS: label_pos,
                 LABEL_ROT: label_rot}
    x_out = sess.run([x_final], feed_dict)[0]

    im_name = "im"

    if FLAGS.cond_scale:
        im_name += "_condscale"

    if FLAGS.cond_shape:
        im_name += "_condshape"

    if FLAGS.cond_pos:
        im_name += "_condpos"

    if FLAGS.cond_rot:
        im_name += "_condrot"

    im_name += ".png"

    x_out_pad, data_try_pad = np.ones((10, 66, 66)), np.ones((10, 66, 66))
    x_out_pad[:, 1:-1, 1:-1] = x_out
    data_try_pad[:, 1:-1, 1:-1] = data_try

    im_output = np.concatenate([x_out_pad, data_try_pad], axis=2).reshape(-1, 66*2)
    impath = osp.join(save_exp_dir, im_name)
    imsave(impath, im_output)
    print("Successfully saved images at {}".format(impath))

def main():
    data = np.load(FLAGS.dsprites_path)['imgs']
    l = latents = np.load(FLAGS.dsprites_path)['latents_values']

    np.random.seed(1)
    idx = np.random.permutation(data.shape[0])

    data = data[idx]
    latents = latents[idx]

    config = tf.ConfigProto()
    sess = tf.Session(config=config)

    # Model 1 will be conditioned on size
    model_size = DspritesNet(num_filters=FLAGS.num_filters, cond_size=True)
    weight_size = model_size.construct_weights('context_0')

    # Model 2 will be conditioned on shape
    model_shape = DspritesNet(num_filters=FLAGS.num_filters, cond_shape=True)
    weight_shape = model_shape.construct_weights('context_1')

    # Model 3 will be conditioned on position
    model_pos = DspritesNet(num_filters=FLAGS.num_filters, cond_pos=True)
    weight_pos = model_pos.construct_weights('context_2')

    # Model 4 will be conditioned on rotation
    model_rot = DspritesNet(num_filters=FLAGS.num_filters, cond_rot=True)
    weight_rot = model_rot.construct_weights('context_3')

    sess.run(tf.global_variables_initializer())
    save_path_size = osp.join(FLAGS.logdir, FLAGS.exp_size, 'model_{}'.format(FLAGS.resume_size))

    v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(0))
    v_map = {(v.name.replace('context_{}'.format(0), 'context_0')[:-2]): v for v in v_list}

    if FLAGS.cond_scale:
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_size)

    save_path_shape = osp.join(FLAGS.logdir, FLAGS.exp_shape, 'model_{}'.format(FLAGS.resume_shape))

    v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(1))
    v_map = {(v.name.replace('context_{}'.format(1), 'context_0')[:-2]): v for v in v_list}

    if FLAGS.cond_shape:
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_shape)


    save_path_pos = osp.join(FLAGS.logdir, FLAGS.exp_pos, 'model_{}'.format(FLAGS.resume_pos))
    v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(2))
    v_map = {(v.name.replace('context_{}'.format(2), 'context_0')[:-2]): v for v in v_list}
    saver = tf.train.Saver(v_map)

    if FLAGS.cond_pos:
        saver.restore(sess, save_path_pos)


    save_path_rot = osp.join(FLAGS.logdir, FLAGS.exp_rot, 'model_{}'.format(FLAGS.resume_rot))
    v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(3))
    v_map = {(v.name.replace('context_{}'.format(3), 'context_0')[:-2]): v for v in v_list}
    saver = tf.train.Saver(v_map)

    if FLAGS.cond_rot:
        saver.restore(sess, save_path_rot)

    X_NOISE = tf.placeholder(shape=(None, 64, 64), dtype=tf.float32)
    LABEL_SIZE = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    LABEL_SHAPE = tf.placeholder(shape=(None, 3), dtype=tf.float32)
    LABEL_POS = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    LABEL_ROT = tf.placeholder(shape=(None, 2), dtype=tf.float32)

    x_mod = X_NOISE

    kvs = {}
    kvs['X_NOISE'] = X_NOISE
    kvs['LABEL_SIZE'] = LABEL_SIZE
    kvs['LABEL_SHAPE'] = LABEL_SHAPE
    kvs['LABEL_POS'] = LABEL_POS
    kvs['LABEL_ROT'] = LABEL_ROT
    kvs['model_size'] = model_size
    kvs['model_shape'] = model_shape
    kvs['model_pos'] = model_pos
    kvs['model_rot'] = model_rot
    kvs['weight_size'] = weight_size
    kvs['weight_shape'] = weight_shape
    kvs['weight_pos'] = weight_pos
    kvs['weight_rot'] = weight_rot

    save_exp_dir = osp.join(FLAGS.savedir, '{}_{}_joint'.format(FLAGS.exp_size, FLAGS.exp_shape))
    if not osp.exists(save_exp_dir):
        os.makedirs(save_exp_dir)


    if FLAGS.task == 'conceptcombine':
        conceptcombine(sess, kvs, data, latents, save_exp_dir)
    elif FLAGS.task == 'labeldiscover':
        labeldiscover(sess, kvs, data, latents, save_exp_dir)
    elif FLAGS.task == 'gentest':
        save_exp_dir = osp.join(FLAGS.savedir, '{}_{}_gen'.format(FLAGS.exp_size, FLAGS.exp_pos))
        if not osp.exists(save_exp_dir):
            os.makedirs(save_exp_dir)

        gentest(sess, kvs, data, latents, save_exp_dir)
    elif FLAGS.task == 'genbaseline':
        save_exp_dir = osp.join(FLAGS.savedir, '{}_{}_gen_baseline'.format(FLAGS.exp_size, FLAGS.exp_pos))
        if not osp.exists(save_exp_dir):
            os.makedirs(save_exp_dir)

        if FLAGS.plot_curve:
            mse_losses = []
            for frac in [i/10 for i in range(11)]:
                mse_loss = genbaseline(sess, kvs, data, latents, save_exp_dir, frac=frac)
                mse_losses.append(mse_loss)
            np.save("mse_baseline_comb.npy", mse_losses)
        else:
            genbaseline(sess, kvs, data, latents, save_exp_dir)



if __name__ == "__main__":
    main()
