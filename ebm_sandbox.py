import tensorflow as tf
import math
from tqdm import tqdm
from hmc import hmc
from tensorflow.python.platform import flags
from torch.utils.data import DataLoader
import torch
from models import ResNet32, ResNet32Large, ResNet32Larger, ResNet32Wider, DspritesNet
from data import Cifar10, Svhn, Cifar100, Textures, Imagenet, DSprites
from utils import optimistic_restore, set_seed
import os.path as osp
import numpy as np
from rl_algs.logger import TensorBoardOutputFormat
from scipy.misc import imsave
import os
import sklearn.metrics as sk
from baselines.common.tf_util import initialize
from scipy.linalg import eig
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

set_seed(5)

flags.DEFINE_string('datasource', 'random', 'default or noise or negative or single')
flags.DEFINE_string('dataset', 'cifar10', 'omniglot or imagenet or omniglotfull or cifar10 or mnist or dsprites')
flags.DEFINE_string('logdir', '/mnt/nfs/yilundu/pot_kmeans/sandbox_cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('task', 'label', 'the task to execute (label: training on the label, anticorrupt: restore salt and pepper noise), boxcorrupt: restore empty portion of image'
                    'or crossclass: change images from one class to another'
                    'or cycleclass: view image change across a label'
                    'or nearestneighbor which returns the nearest images in the test set'
                    'or labelfinetune to train a model accuracy'
                    'or latent to traverse the latent through energy')

flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('data_workers', 5, 'Number of different data workers to load data in parallel')
flags.DEFINE_integer('batch_size', 32, 'Size of inputs')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('bn', False, 'Whether to use batch normalization or not')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_bias', True, 'Whether to use bias in convolution')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_bool('train', True, 'Whether to train or test network')
flags.DEFINE_bool('single', False, 'whether to use one sample to debug')
flags.DEFINE_bool('cclass', True, 'whether to use a conditional model (required for task label)')
flags.DEFINE_integer('num_steps', 20, 'number of steps to optimize the label')
flags.DEFINE_integer('pgd', 0, 'number of steps project gradient descent to run')
flags.DEFINE_integer('lnorm', -1, 'lnorm infinity is -1, ')
flags.DEFINE_float('step_lr', 10.0, 'step size for updates on label')
flags.DEFINE_float('proj_norm', 0.0, 'Maximum change of input images')
flags.DEFINE_bool('large_model', False, 'Whether to use a large model')
flags.DEFINE_bool('larger_model', False, 'Whether to use a large model')
flags.DEFINE_bool('wider_model', False, 'Whether to use a large model')
flags.DEFINE_bool('svhn', False, 'Whether to test on SVHN')
flags.DEFINE_bool('svhnmix', False, 'Whether to test mix on SVHN')
flags.DEFINE_bool('cifar100mix', False, 'Whether to test mix on CIFAR100')
flags.DEFINE_bool('texturemix', False, 'Whether to test mix on CIFAR100')
flags.DEFINE_bool('randommix', False, 'Whether to test mix on CIFAR100')
flags.DEFINE_bool('groupsort', False, 'Whether to test mix on CIFAR100')
flags.DEFINE_bool('hmc', False, 'Use HMC for cross class sampling')
flags.DEFINE_bool('labelgrid', False, 'Make a grid of labels')
flags.DEFINE_bool('proj_cclass', False, 'Projection conditional')

# Conditions on which models to use
flags.DEFINE_bool('cond_pos', True, 'whether to condition on position')
flags.DEFINE_bool('cond_rot', True, 'whether to condition on rotation')
flags.DEFINE_bool('cond_shape', True, 'whether to condition on shape')
flags.DEFINE_bool('cond_size', True, 'whether to condition on scale')

FLAGS = flags.FLAGS

def rescale_im(im):
    im = np.clip(im, 0, 1)
    return np.round(im * 255).astype(np.uint8)

def label(dataloader, test_dataloader, target_vars, sess, l1val=8, l2val=40):
    X = target_vars['X']
    Y = target_vars['Y']
    Y_GT = target_vars['Y_GT']
    accuracy = target_vars['accuracy']
    train_op = target_vars['train_op']
    l1_norm = target_vars['l1_norm']
    l2_norm = target_vars['l2_norm']

    label_init = np.random.uniform(0, 1, (FLAGS.batch_size, 10))
    label_init = label_init / label_init.sum(axis=1, keepdims=True)

    label_init = np.tile(np.eye(10)[None :, :], (FLAGS.batch_size, 1, 1))
    label_init = np.reshape(label_init, (-1, 10))

    for i in range(1):
        emp_accuracies = []

        for data_corrupt, data, label_gt in tqdm(test_dataloader):
            feed_dict = {X: data, Y_GT: label_gt, Y: label_init, l1_norm: l1val, l2_norm: l2val}
            emp_accuracy = sess.run([accuracy], feed_dict)
            emp_accuracies.append(emp_accuracy)
            print(np.array(emp_accuracies).mean())

        print("Received total accuracy of {} for li of {} and l2 of {}".format(np.array(emp_accuracies).mean(), l1val, l2val))

    return np.array(emp_accuracies).mean()


def labelfinetune(dataloader, test_dataloader, target_vars, sess, savedir, saver, l1val=8, l2val=40):
    X = target_vars['X']
    Y = target_vars['Y']
    Y_GT = target_vars['Y_GT']
    accuracy = target_vars['accuracy']
    train_op = target_vars['train_op']
    l1_norm = target_vars['l1_norm']
    l2_norm = target_vars['l2_norm']

    label_init = np.random.uniform(0, 1, (FLAGS.batch_size, 10))
    label_init = label_init / label_init.sum(axis=1, keepdims=True)

    label_init = np.tile(np.eye(10)[None :, :], (FLAGS.batch_size, 1, 1))
    label_init = np.reshape(label_init, (-1, 10))

    itr = 0

    if FLAGS.train:
        for i in range(1):
            for data_corrupt, data, label_gt in tqdm(dataloader):
                feed_dict = {X: data, Y_GT: label_gt, Y: label_init}
                acc, _ = sess.run([accuracy, train_op], feed_dict)

                itr += 1

                if itr % 10 == 0:
                    print(acc)

        saver.save(sess, osp.join(savedir, "model_supervised"))

    saver.restore(sess, osp.join(savedir, "model_supervised"))


    for i in range(1):
        emp_accuracies = []

        for data_corrupt, data, label_gt in tqdm(test_dataloader):
            feed_dict = {X: data, Y_GT: label_gt, Y: label_init, l1_norm: l1val, l2_norm: l2val}
            emp_accuracy = sess.run([accuracy], feed_dict)
            emp_accuracies.append(emp_accuracy)
            print(np.array(emp_accuracies).mean())


        print("Received total accuracy of {} for li of {} and l2 of {}".format(np.array(emp_accuracies).mean(), l1val, l2val))

    return np.array(emp_accuracies).mean()


def energyeval(dataloader, test_dataloader, target_vars, sess):
    X = target_vars['X']
    Y_GT = target_vars['Y_GT']
    energy = target_vars['energy']
    energy_end = target_vars['energy_end']

    test_energies = []
    train_energies = []
    for data_corrupt, data, label_gt in tqdm(test_dataloader):
        feed_dict = {X: data, Y_GT: label_gt}
        test_energy = sess.run([energy], feed_dict)[0]
        test_energies.extend(list(test_energy))

    for data_corrupt, data, label_gt in tqdm(dataloader):
        feed_dict = {X: data, Y_GT: label_gt}
        train_energy = sess.run([energy], feed_dict)[0]
        train_energies.extend(list(train_energy))

    print(len(train_energies))
    print(len(test_energies))

    print("Train energies of {} with std {}".format(np.mean(train_energies), np.std(train_energies)))
    print("Test energies of {} with std {}".format(np.mean(test_energies), np.std(test_energies)))

    np.save("train_ebm.npy", train_energies)
    np.save("test_ebm.npy", test_energies)


def energyevalmix(dataloader, test_dataloader, target_vars, sess):
    X = target_vars['X']
    Y_GT = target_vars['Y_GT']
    energy = target_vars['energy']

    if FLAGS.svhnmix:
        dataset = Svhn(train=False)
        test_dataloader_val = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, shuffle=True, drop_last=False)
        test_iter = iter(test_dataloader_val)
    elif FLAGS.cifar100mix:
        dataset = Cifar100(train=False)
        test_dataloader_val = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, shuffle=True, drop_last=False)
        test_iter = iter(test_dataloader_val)
    elif FLAGS.texturemix:
        dataset = Textures()
        test_dataloader_val = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, shuffle=True, drop_last=False)
        test_iter = iter(test_dataloader_val)

    probs = []
    labels = []
    negs = []
    pos = []
    for data_corrupt, data, label_gt in tqdm(test_dataloader):
        data = data.numpy()
        data_corrupt = data_corrupt.numpy()
        if FLAGS.svhnmix:
            _, data_mix, _ = test_iter.next()
        elif FLAGS.cifar100mix:
            _, data_mix, _ = test_iter.next()
        elif FLAGS.texturemix:
            _, data_mix, _ = test_iter.next()
        elif FLAGS.randommix:
            data_mix = np.random.randn(FLAGS.batch_size, 32, 32, 3) * 0.5 + 0.5
        else:
            data_idx = np.concatenate([np.arange(1, data.shape[0]), [0]])
            data_other = data[data_idx]
            data_mix = (data + data_other) / 2

        data_mix = data_mix[:data.shape[0]]

        if FLAGS.cclass:
            # It's unfair to take a random class
            label_gt= np.tile(np.eye(10), (data.shape[0], 1, 1))
            label_gt = label_gt.reshape(data.shape[0] * 10, 10)
            data_mix = np.tile(data_mix[:, None, :, :, :], (1, 10, 1, 1, 1))
            data = np.tile(data[:, None, :, :, :], (1, 10, 1, 1, 1))

            data_mix = data_mix.reshape(-1, 32, 32, 3)
            data = data.reshape(-1, 32, 32, 3)


        feed_dict = {X: data, Y_GT: label_gt}
        feed_dict_neg = {X: data_mix, Y_GT: label_gt}

        pos_energy = sess.run([energy], feed_dict)[0]
        neg_energy = sess.run([energy], feed_dict_neg)[0]

        if FLAGS.cclass:
            pos_energy = pos_energy.reshape(-1, 10).min(axis=1)
            neg_energy = neg_energy.reshape(-1, 10).min(axis=1)

        probs.extend(list(-1*pos_energy))
        probs.extend(list(-1*neg_energy))
        pos.extend(list(-1*pos_energy))
        negs.extend(list(-1*neg_energy))
        labels.extend([1]*pos_energy.shape[0])
        labels.extend([0]*neg_energy.shape[0])

    pos, negs = np.array(pos), np.array(negs)
    np.save("pos.npy", pos)
    np.save("neg.npy", negs)
    auroc = sk.roc_auc_score(labels, probs)
    print("Roc score of {}".format(auroc))


def anticorrupt(dataloader, weights, model, target_vars, logdir, sess):
    X, Y_GT, X_final = target_vars['X'], target_vars['Y_GT'], target_vars['X_final']
    for data_corrupt, data, label_gt in tqdm(dataloader):
        data, label_gt = data.numpy(), label_gt.numpy()

        noise = np.random.uniform(0, 1, size=[data.shape[0], data.shape[1], data.shape[2]])
        low_mask = noise < 0.05
        high_mask = (noise > 0.05) & (noise < 0.1)

        print(high_mask.shape)

        data_corrupt = data.copy()
        data_corrupt[low_mask] = 0.1
        data_corrupt[high_mask] = 0.9
        data_corrupt_init = data_corrupt

        for i in range(5):
            feed_dict = {X: data_corrupt, Y_GT: label_gt}
            data_corrupt = sess.run([X_final], feed_dict)[0]

        data_uncorrupt = data_corrupt
        data_corrupt, data_uncorrupt, data = rescale_im(data_corrupt_init), rescale_im(data_uncorrupt), rescale_im(data)

        panel_im = np.zeros((32*20, 32*3, 3)).astype(np.uint8)

        for i in range(20):
            panel_im[32*i:32*i+32, :32] = data_corrupt[i]
            panel_im[32*i:32*i+32, 32:64] = data_uncorrupt[i]
            panel_im[32*i:32*i+32, 64:] = data[i]

        imsave(osp.join(logdir, "anticorrupt.png"), panel_im)
        assert False


def boxcorrupt(test_dataloader, dataloader, weights, model, target_vars, logdir, sess):
    X, Y_GT, X_final = target_vars['X'], target_vars['Y_GT'], target_vars['X_final']
    eval_im = 10000

    data_diff = []
    for data_corrupt, data, label_gt in tqdm(dataloader):
        data, label_gt = data.numpy(), label_gt.numpy()
        data_uncorrupts =  []

        data_corrupt = data.copy()
        data_corrupt[:, 16:, :] = np.random.uniform(0, 1, (FLAGS.batch_size, 16, 32, 3))

        data_corrupt_init = data_corrupt

        for j in range(10):
            feed_dict = {X: data_corrupt, Y_GT: label_gt}
            data_corrupt = sess.run([X_final], feed_dict)[0]

        val = np.mean(np.square(data_corrupt - data), axis=(1, 2, 3))
        data_diff.extend(list(val))

        if len(data_diff) > eval_im:
            break

    print("Mean {} and std {} for train dataloader".format(np.mean(data_diff), np.std(data_diff)))

    np.save("data_diff_train_image.npy", data_diff)

    data_diff = []

    for data_corrupt, data, label_gt in tqdm(test_dataloader):
        data, label_gt = data.numpy(), label_gt.numpy()
        data_uncorrupts =  []

        data_corrupt = data.copy()
        data_corrupt[:, 16:, :] = np.random.uniform(0, 1, (FLAGS.batch_size, 16, 32, 3))

        data_corrupt_init = data_corrupt

        for j in range(10):
            feed_dict = {X: data_corrupt, Y_GT: label_gt}
            data_corrupt = sess.run([X_final], feed_dict)[0]

        data_diff.extend(list(np.mean(np.square(data_corrupt - data), axis=(1, 2, 3))))

        if len(data_diff) > eval_im:
            break

    print("Mean {} and std {} for test dataloader".format(np.mean(data_diff), np.std(data_diff)))

    np.save("data_diff_test_image.npy", data_diff)


def crossclass(dataloader, weights, model, target_vars, logdir, sess):
    X, Y_GT, X_mods = target_vars['X'], target_vars['Y_GT'], target_vars['X_mods']
    for data_corrupt, data, label_gt in tqdm(dataloader):
        data, label_gt = data.numpy(), label_gt.numpy()
        data_corrupt = data.copy()
        data_corrupt[1:] = data_corrupt[0:-1]
        data_corrupt[0] = data[-1]

        feed_dict = {X: data_corrupt, Y_GT: label_gt}
        data_mods = sess.run(X_mods, feed_dict)

        data_corrupt, data = rescale_im(data_corrupt), rescale_im(data)

        data_mods = [rescale_im(data_mod) for data_mod in data_mods]

        panel_im = np.zeros((32*20, 32*(len(data_mods) + 2), 3)).astype(np.uint8)

        for i in range(20):
            panel_im[32*i:32*i+32, :32] = data_corrupt[i]

            for j in range(len(data_mods)):
                panel_im[32*i:32*i+32, 32*(j+1):32*(j+2)] = data_mods[j][i]

            panel_im[32*i:32*i+32, -32:] = data[i]

        imsave(osp.join(logdir, "crossclass.png"), panel_im)
        assert False


def cycleclass(dataloader, weights, model, target_vars, logdir, sess):
    # X, Y_GT, X_final, X_targ = target_vars['X'], target_vars['Y_GT'], target_vars['X_final'], target_vars['X_targ']
    X, Y_GT, X_final = target_vars['X'], target_vars['Y_GT'], target_vars['X_final']
    for data_corrupt, data, label_gt in tqdm(dataloader):
        data, label_gt = data.numpy(), label_gt.numpy()
        data_corrupt = data_corrupt.numpy()


        data_mods = []
        x_curr = data_corrupt
        x_target = np.random.uniform(0, 1, data_corrupt.shape)
        # x_target = np.tile(x_target, (1, 32, 32, 1))


        for i in range(20):
            feed_dict = {X: x_curr, Y_GT: label_gt}
            x_curr_new = sess.run(X_final, feed_dict)
            x_curr = x_curr_new
            data_mods.append(x_curr_new)

            if i > 30:
                x_target = np.random.uniform(0, 1, data_corrupt.shape)

        data_corrupt, data = rescale_im(data_corrupt), rescale_im(data)

        data_mods = [rescale_im(data_mod) for data_mod in data_mods]

        panel_im = np.zeros((32*100, 32*(len(data_mods) + 2), 3)).astype(np.uint8)

        for i in range(100):
            panel_im[32*i:32*i+32, :32] = data_corrupt[i]

            for j in range(len(data_mods)):
                panel_im[32*i:32*i+32, 32*(j+1):32*(j+2)] = data_mods[j][i]

            panel_im[32*i:32*i+32, -32:] = data[i]

        imsave(osp.join(logdir, "cycleclass.png"), panel_im)
        assert False


def democlass(dataloader, weights, model, target_vars, logdir, sess):
    X, Y_GT, X_final = target_vars['X'], target_vars['Y_GT'], target_vars['X_final']
    panel_im = np.zeros((5*32, 10*32, 3)).astype(np.uint8)
    for i in range(10):
        data_corrupt = np.random.uniform(0, 1, (5, 32, 32, 3))
        label_gt = np.tile(np.eye(10)[i:i+1], (5, 1))

        feed_dict = {X: data_corrupt, Y_GT: label_gt}
        x_final = sess.run([X_final], feed_dict)[0]

        x_final = rescale_im(x_final)

        row = i // 2
        col = i % 2

        start_idx = col * 32 * 5
        row_idx = row * 32

        for j in range(5):
            panel_im[row_idx:row_idx+32, start_idx+j*32:start_idx+(j+1) * 32] = x_final[j]

    imsave(osp.join(logdir, "democlass.png"), panel_im)


def construct_finetune_label(weight, X, Y, Y_GT, model, target_vars):
    l1_norm = tf.placeholder(shape=(), dtype=tf.float32)
    l2_norm = tf.placeholder(shape=(), dtype=tf.float32)

    def compute_logit(X, stop_grad=False, num_steps=0):
        batch_size = tf.shape(X)[0]
        X = tf.reshape(X, (batch_size, 1, 32, 32, 3))
        X = tf.reshape(tf.tile(X, (1, 10, 1, 1, 1)), (batch_size * 10, 32, 32, 3))
        Y_new = tf.reshape(Y, (batch_size*10, 10))

        X_min = X - 8 / 255.
        X_max = X + 8 / 255.

        for i in range(num_steps):
            X = X + tf.random_normal(tf.shape(X), mean=0.0, stddev=0.005)

            energy_noise = model.forward(X, weights, label=Y, reuse=True)
            x_grad = tf.gradients(energy_noise, [X])[0]


            if FLAGS.proj_norm != 0.0:
                x_grad = tf.clip_by_value(x_grad, -FLAGS.proj_norm, FLAGS.proj_norm)

            X = X - FLAGS.step_lr * x_grad
            X = tf.maximum(tf.minimum(X, X_max), X_min)

        energy = model.forward(X, weight, label=Y_new)
        energy = -tf.reshape(energy, (batch_size, 10))

        if stop_grad:
            energy = tf.stop_gradient(energy)

        return energy

    for i in range(FLAGS.pgd):
        if FLAGS.train:
            break

        print("Constructed loop {} of pgd attack".format(i))
        X_init = X
        if i == 0:
            X = X + tf.to_float(tf.random_uniform(tf.shape(X), minval=-8, maxval=9, dtype=tf.int32)) / 255.

        logit = compute_logit(X)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_GT, logits=logit)

        x_grad = tf.sign(tf.gradients(loss, [X])[0]) / 255.
        X = X + 2 * x_grad

        if FLAGS.lnorm == -1:
            X = tf.maximum(tf.minimum(X, X_max), X_min)
        elif FLAGS.lnorm == 2:
            X = X_init + tf.clip_by_norm(X - X_init, l2_norm / 255., axes=[1, 2, 3])


    energy = compute_logit(X, num_steps=0)
    logits = energy
    labels = tf.argmax(Y_GT, axis=1)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_GT, logits=logits)


    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(loss)
    accuracy = tf.contrib.metrics.accuracy(tf.argmax(logits, axis=1), labels)

    target_vars['accuracy'] = accuracy
    target_vars['train_op'] = train_op
    target_vars['l1_norm'] = l1_norm
    target_vars['l2_norm'] = l2_norm


def construct_latent(weights, X, Y_GT, model, target_vars):

    eps = 0.001
    X_init = X[0:1]
    e_pos = e_pos_base = model.forward(X_init, weights, label=Y_GT)
    hessian = tf.hessians(e_pos, X_init)

    hessian = tf.reshape(hessian, (1, 64*64, 64*64))[0]

    e, v = tf.linalg.eigh(hessian)

    var_scale = 0.1
    n = 3
    xs = []

    for i in range(n):
        var = tf.reshape(v[:, i], (1, 64, 64))
        X_plus = X_init - var_scale * var
        X_min = X_init + var_scale * var

        xs.extend([X_plus, X_min])

    x_stack = tf.stack(xs, axis=0)

    e_pos_hess_modify = model.forward(x_stack, weights, label=Y_GT)

    for i in range(1):
        x_stack = x_stack + tf.random_normal(tf.shape(x_stack), mean=0.0, stddev=0.005)
        e_pos = model.forward(x_stack, weights, label=Y_GT)

        x_grad = tf.gradients(e_pos, [x_stack])[0]
        x_stack = x_stack - 4*FLAGS.step_lr * x_grad

        x_stack = tf.clip_by_value(x_stack, 0, 1)

    x_mods = tf.split(X, 6)

    eigs = []
    for j in range(6):
        x_mod = x_mods[j]
        e_pos = model.forward(x_mod, weights, label=Y_GT)
        hessian = tf.hessians(e_pos, x_mod)
        hessian = tf.reshape(hessian, (1, 64*64, 64*64))[0]
        e, v = tf.linalg.eigh(hessian)

        idx = j // 2
        var = tf.reshape(v[:, idx], (1, 64, 64))

        if j % 2 == 1:
            x_mod = x_mod + var_scale * var
            eigs.append(var)
        else:
            x_mod = x_mod - var_scale * var
            eigs.append(-var)

        x_mod = tf.clip_by_value(x_mod, 0, 1)
        x_mods[j] = x_mod

    x_mods_stack = tf.stack(x_mods, axis=0)

    eigs_stack = tf.stack(eigs, axis=0)
    energys = []

    for i in range(1):
        x_mods_stack = x_mods_stack + tf.random_normal(tf.shape(x_mods_stack), mean=0.0, stddev=0.005)
        e_pos = model.forward(x_mods_stack, weights, label=Y_GT)

        x_grad = tf.gradients(e_pos, [x_mods_stack])[0]
        x_mods_stack = x_mods_stack - 4*FLAGS.step_lr * x_grad
        # x_mods_stack = x_mods_stack + 0.1 * eigs_stack

        x_mods_stack = tf.clip_by_value(x_mods_stack, 0, 1)

        energys.append(e_pos)

    x_refine = x_mods_stack
    es = tf.stack(energys, axis=0)

    target_vars['hessian'] = hessian
    target_vars['e'] = e
    target_vars['v'] = v
    target_vars['x_stack'] = x_stack
    target_vars['x_refine'] = x_refine
    target_vars['es'] = es
    target_vars['e_base'] = e_pos_base
    target_vars['e_pos_hessian'] = e_pos_hess_modify


def latent(test_dataloader, weights, model, target_vars, sess):
    X = target_vars['X']
    Y_GT = target_vars['Y_GT']
    hessian = target_vars['hessian']
    e = target_vars['e']
    v = target_vars['v']
    x_stack = target_vars['x_stack']
    x_refine = target_vars['x_refine']
    es = target_vars['es']
    e_pos_base = target_vars['e_base']
    e_pos_hess_modify = target_vars['e_pos_hessian']

    data_corrupt, data, label_gt = iter(test_dataloader).next()
    data = data.numpy()
    x_init = np.tile(data[0:1], (6, 1, 1))
    x_mod, e_pos, e_pos_hess = sess.run([x_stack, e_pos_base, e_pos_hess_modify], {X: data})
    print("Value of original starting image: ", e_pos)
    print("Value of energy of hessian: ", e_pos_hess)
    x_mod = x_mod.squeeze()

    n = 5
    x_mod_list = [x_init, x_mod]

    for i in range(n):
        x_mod, evals = sess.run([x_refine, es], {X: x_mod})
        x_mod = x_mod.squeeze()
        x_mod_list.append(x_mod)
        print("Value of energies after evaluation: ", evals)

    x_mod_list = x_mod_list[:]


    series_xmod = np.stack(x_mod_list, axis=1)
    series_header = np.tile(data[0:1, None, :, :], (1, len(x_mod_list), 1, 1))

    series_total = np.concatenate([series_header, series_xmod], axis=0)

    series_total_full = np.ones((*series_total.shape[:-2], 66, 66))

    series_total_full[:, :, 1:-1, 1:-1] = series_total

    series_total = series_total_full

    series_total = series_total.transpose((0, 2, 1, 3)).reshape((-1, len(x_mod_list)*66))
    im_total = rescale_im(series_total)
    imsave("latent_comb.png", im_total)


def construct_label(weights, X, Y, Y_GT, model, target_vars):
    # for i in range(FLAGS.num_steps):
    #     Y = Y + tf.random_normal(tf.shape(Y), mean=0.0, stddev=0.03)
    #     e = model.forward(X, weights, label=Y)

    #     Y_grad = tf.clip_by_value(tf.gradients(e, [Y])[0],  -1, 1)
    #     Y = Y - 0.1 * Y_grad
    #     Y = tf.clip_by_value(Y, 0, 1)

    #     Y = Y / tf.reduce_sum(Y, axis=[1], keepdims=True)

    e_bias =  tf.get_variable('e_bias', shape=10, initializer=tf.initializers.zeros())
    l1_norm = tf.placeholder(shape=(), dtype=tf.float32)
    l2_norm = tf.placeholder(shape=(), dtype=tf.float32)

    def compute_logit(X, stop_grad=False, num_steps=0):
        batch_size = tf.shape(X)[0]
        X = tf.reshape(X, (batch_size, 1, 32, 32, 3))
        X = tf.reshape(tf.tile(X, (1, 10, 1, 1, 1)), (batch_size * 10, 32, 32, 3))
        Y_new = tf.reshape(Y, (batch_size*10, 10))

        X_min = X - 8 / 255.
        X_max = X + 8 / 255.

        for i in range(num_steps):
            X = X + tf.random_normal(tf.shape(X), mean=0.0, stddev=0.005)

            energy_noise = model.forward(X, weights, label=Y, reuse=True)
            x_grad = tf.gradients(energy_noise, [X])[0]


            if FLAGS.proj_norm != 0.0:
                x_grad = tf.clip_by_value(x_grad, -FLAGS.proj_norm, FLAGS.proj_norm)

            X = X - FLAGS.step_lr * x_grad
            X = tf.maximum(tf.minimum(X, X_max), X_min)

        energy = model.forward(X, weights, label=Y_new)
        energy = -tf.reshape(energy, (batch_size, 10))

        if stop_grad:
            energy = tf.stop_gradient(energy)

        return energy


    # eps_norm = 30
    X_min = X - l1_norm / 255.
    X_max = X + l1_norm / 255.

    for i in range(FLAGS.pgd):
        print("Constructed loop {} of pgd attack".format(i))
        X_init = X
        if i == 0:
            X = X + tf.to_float(tf.random_uniform(tf.shape(X), minval=-8, maxval=9, dtype=tf.int32)) / 255.

        logit = compute_logit(X)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_GT, logits=logit)

        x_grad = tf.sign(tf.gradients(loss, [X])[0]) / 255.
        X = X + 2 * x_grad

        if FLAGS.lnorm == -1:
            X = tf.maximum(tf.minimum(X, X_max), X_min)
        elif FLAGS.lnorm == 2:
            X = X_init + tf.clip_by_norm(X - X_init, l2_norm / 255., axes=[1, 2, 3])

    energy_stopped = compute_logit(X, stop_grad=True, num_steps=FLAGS.num_steps) + e_bias

    # # Y = tf.Print(Y, [Y])
    labels = tf.argmax(Y_GT, axis=1)
    # max_z = tf.argmax(energy_stopped, axis=1)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_GT, logits=energy_stopped)
    optimizer = tf.train.AdamOptimizer(1e-2)
    train_op = optimizer.minimize(loss)

    accuracy = tf.contrib.metrics.accuracy(tf.argmax(energy_stopped, axis=1), labels)
    target_vars['accuracy'] = accuracy
    target_vars['train_op'] = train_op
    target_vars['l1_norm'] = l1_norm
    target_vars['l2_norm'] = l2_norm


def construct_energy(weights, X, Y, Y_GT, model, target_vars):
    energy = model.forward(X, weights, label=Y_GT)

    for i in range(FLAGS.num_steps):
        X = X + tf.random_normal(tf.shape(X), mean=0.0, stddev=0.005)

        energy_noise = model.forward(X, weights, label=Y_GT, reuse=True)
        x_grad = tf.gradients(energy_noise, [X])[0]

        if FLAGS.proj_norm != 0.0:
            x_grad = tf.clip_by_value(x_grad, -FLAGS.proj_norm, FLAGS.proj_norm)

        X = X - FLAGS.step_lr * x_grad
        X = tf.clip_by_value(X, 0, 1)


    target_vars['energy'] = energy
    target_vars['energy_end'] = energy_noise


def construct_steps(weights, X, Y_GT, model, target_vars):
    n = 50
    scale_fac = 1.0

    # if FLAGS.task == 'cycleclass':
    #     scale_fac = 10.0

    X_mods = []
    X = tf.identity(X)

    mask = np.zeros((1, 32, 32, 3))

    if FLAGS.task == "boxcorrupt":
        mask[:, 16:, :, :] = 1
    else:
        mask[:, :, :, :] = 1

    mask = tf.Variable(tf.convert_to_tensor(mask, dtype=tf.float32), trainable=False)

    # X_targ = tf.placeholder(shape=(None, 32, 32, 3), dtype = tf.float32)

    for i in range(FLAGS.num_steps):
        X_old = X
        X = X + tf.random_normal(tf.shape(X), mean=0.0, stddev=0.005*scale_fac) * mask

        energy_noise = model.forward(X, weights, label=Y_GT, reuse=True)
        x_grad = tf.gradients(energy_noise, [X])[0]

        if FLAGS.proj_norm != 0.0:
            x_grad = tf.clip_by_value(x_grad, -FLAGS.proj_norm, FLAGS.proj_norm)

        X = X - FLAGS.step_lr * x_grad * scale_fac * mask
        X = tf.clip_by_value(X, 0, 1)

        if i % n == (n-1):
            X_mods.append(X)

        print("Constructing step {}".format(i))

    target_vars['X_final'] = X
    target_vars['X_mods'] = X_mods


def construct_hmc_steps(weights, X, Y_GT, model, target_vars):
    n = 50
    scale_fac = 1.0

    # if FLAGS.task == 'cycleclass':
    #     scale_fac = 10.0

    X_mods = []
    X = tf.identity(X)

    for i in range(FLAGS.num_steps):
        p = tf.random_normal(tf.shape(X), mean=0.0, stddev=0.0001)
        for j in range(10):
            energy_noise = model.forward(X, weights, label=Y_GT, reuse=True)
            x_grad = tf.gradients(energy_noise, [X])[0]
            p = p - FLAGS.step_lr * x_grad / 2
            X = X - FLAGS.step_lr * p

            if i % n == (n-1):
                X_mods.append(X)

            print("Constructing step {}".format(i))

    target_vars['X_final'] = X
    target_vars['X_mods'] = X_mods


def nearest_neighbor(dataset, sess, target_vars, logdir):
    X = target_vars['X']
    Y_GT = target_vars['Y_GT']
    x_final = target_vars['X_final']

    noise = np.random.uniform(0, 1, size=[10, 32, 32, 3])
    # label = np.random.randint(0, 10, size=[10])
    label = np.eye(10)

    coarse = noise

    for i in range(10):
        x_new = sess.run([x_final], {X:coarse, Y_GT:label})[0]
        coarse = x_new

    x_new_dense = x_new.reshape(10, 1, 32*32*3)
    dataset_dense = dataset.reshape(1, 50000, 32*32*3)

    diff = np.square(x_new_dense - dataset_dense).sum(axis=2)
    diff_idx = np.argsort(diff, axis=1)

    panel = np.zeros((32*10, 32*6, 3))

    dataset_rescale = rescale_im(dataset)
    x_new_rescale = rescale_im(x_new)

    for i in range(10):
        panel[i*32:i*32+32, :32] = x_new_rescale[i]
        for j in range(5):
            panel[i*32:i*32+32, 32*j+32:32*j+64] = dataset_rescale[diff_idx[i, j]]

    imsave(osp.join(logdir, "nearest.png"), panel)


def main():

    if FLAGS.dataset == "cifar10":
        dataset = Cifar10(train=True, noise=False)
        test_dataset = Cifar10(train=False, noise=False)
    else:
        dataset = Imagenet(train=True)
        test_dataset = Imagenet(train=False)

    if FLAGS.svhn:
        dataset = Svhn(train=True)
        test_dataset = Svhn(train=False)

    if FLAGS.task == 'latent':
        dataset = DSprites()
        test_dataset = dataset

    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, shuffle=True, drop_last=True)

    hidden_dim = 128

    if FLAGS.large_model:
        model = ResNet32Large(num_filters=hidden_dim)
    elif FLAGS.larger_model:
        model = ResNet32Larger(num_filters=hidden_dim)
    elif FLAGS.wider_model:
        model = ResNet32Wider(num_filters=196, train=False)
    else:
        model = ResNet32(num_filters=hidden_dim)

    if FLAGS.task  == 'latent':
        model = DspritesNet()

    weights = model.construct_weights('context_{}'.format(0))

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Model has a total of {} parameters".format(total_parameters))

    config = tf.ConfigProto()
    sess = tf.InteractiveSession()

    if FLAGS.task == 'latent':
        X = tf.placeholder(shape=(None, 64, 64), dtype = tf.float32)
    else:
        X = tf.placeholder(shape=(None, 32, 32, 3), dtype = tf.float32)

    if FLAGS.dataset == "cifar10":
        Y = tf.placeholder(shape=(None, 10), dtype = tf.float32)
        Y_GT = tf.placeholder(shape=(None, 10), dtype = tf.float32)
    elif FLAGS.dataset == "imagenet":
        Y = tf.placeholder(shape=(None, 1000), dtype = tf.float32)
        Y_GT = tf.placeholder(shape=(None, 1000), dtype = tf.float32)

    target_vars = {'X': X, 'Y': Y, 'Y_GT': Y_GT}

    if FLAGS.task == 'label':
        construct_label(weights, X, Y, Y_GT, model, target_vars)
    elif FLAGS.task == 'labelfinetune':
        construct_finetune_label(weights, X, Y, Y_GT, model, target_vars)
    elif FLAGS.task == 'energyeval' or FLAGS.task == 'mixenergy':
        construct_energy(weights, X, Y, Y_GT, model, target_vars)
    elif FLAGS.task == 'anticorrupt' or FLAGS.task == 'boxcorrupt' or FLAGS.task == 'crossclass' or FLAGS.task == 'cycleclass' or FLAGS.task == 'democlass' or FLAGS.task == 'nearestneighbor':
        if FLAGS.hmc:
            construct_hmc_steps(weights, X, Y_GT, model, target_vars)
        else:
            construct_steps(weights, X, Y_GT, model, target_vars)
    elif FLAGS.task == 'latent':
        construct_latent(weights, X, Y_GT, model, target_vars)

    sess.run(tf.global_variables_initializer())
    saver = loader = tf.train.Saver(max_to_keep=10)
    savedir = osp.join('/mnt/nfs/yilundu/pot_kmeans/cachedir', FLAGS.exp)
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)

    initialize()
    if FLAGS.resume_iter != -1:
        model_file = osp.join(savedir, 'model_{}'.format(FLAGS.resume_iter))
        resume_itr = FLAGS.resume_iter

        if FLAGS.task == 'label' or FLAGS.task == 'boxcorrupt' or FLAGS.task == 'labelfinetune' or FLAGS.task == "energyeval":
            optimistic_restore(sess, model_file)
            # saver.restore(sess, model_file)
        else:
            # optimistic_restore(sess, model_file)
            saver.restore(sess, model_file)

    if FLAGS.task == 'label':
        if FLAGS.labelgrid:
            vals = []
            if FLAGS.lnorm == -1:
                for i in range(31):
                    accuracies = label(dataloader, test_dataloader, target_vars, sess, l1val=i)
                    vals.append(accuracies)
            elif FLAGS.lnorm == 2:
                for i in range(0, 100, 5):
                    accuracies = label(dataloader, test_dataloader, target_vars, sess, l2val=i)
                    vals.append(accuracies)

            np.save("result_{}_{}.npy".format(FLAGS.lnorm, FLAGS.exp), vals)
        else:
            label(dataloader, test_dataloader, target_vars, sess)
    elif FLAGS.task == 'labelfinetune':
        labelfinetune(dataloader, test_dataloader, target_vars, sess, savedir, saver)
    elif FLAGS.task == 'energyeval':
        energyeval(dataloader, test_dataloader, target_vars, sess)
    elif FLAGS.task == 'mixenergy':
        energyevalmix(dataloader, test_dataloader, target_vars, sess)
    elif FLAGS.task == 'anticorrupt':
        anticorrupt(test_dataloader, weights, model, target_vars, logdir, sess)
    elif FLAGS.task == 'boxcorrupt':
        # boxcorrupt(test_dataloader, weights, model, target_vars, logdir, sess)
        boxcorrupt(test_dataloader, dataloader, weights, model, target_vars, logdir, sess)
    elif FLAGS.task == 'crossclass':
        crossclass(test_dataloader, weights, model, target_vars, logdir, sess)
    elif FLAGS.task == 'cycleclass':
        cycleclass(test_dataloader, weights, model, target_vars, logdir, sess)
    elif FLAGS.task == 'democlass':
        democlass(test_dataloader, weights, model, target_vars, logdir, sess)
    elif FLAGS.task == 'nearestneighbor':
        # print(dir(dataset))
        # print(type(dataset))
        nearest_neighbor(dataset.data.train_data / 255, sess, target_vars, logdir)
    elif FLAGS.task == 'latent':
        latent(test_dataloader, weights, model, target_vars, sess)


if __name__ == "__main__":
    main()
