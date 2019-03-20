import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
from models import ResNet32, ResNet32Large, ResNet32Larger, ResNet32Wider, ResNet128
import os.path as osp
import os
from utils import optimistic_restore, remap_restore, optimistic_remap_restore
from tqdm import tqdm
import random
from scipy.misc import imsave
from data import Cifar10, Svhn, Cifar100, Textures, Imagenet, TFImagenetLoader
from torch.utils.data import DataLoader
from baselines.common.tf_util import initialize

import horovod.tensorflow as hvd
hvd.init()

from inception import get_inception_score
from fid import get_fid_score

flags.DEFINE_string('logdir', '/mnt/nfs/yilundu/ebm_code_release/cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_bool('cclass', False, 'whether to condition on class')

# Architecture settings
flags.DEFINE_bool('bn', False, 'Whether to use batch normalization or not')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_bias', True, 'Whether to use bias in convolution')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_float('step_lr', 10.0, 'Size of steps for gradient descent')
flags.DEFINE_integer('num_steps', 20, 'number of steps to optimize the label')
flags.DEFINE_float('proj_norm', 0.05, 'Maximum change of input images')
flags.DEFINE_integer('batch_size', 512, 'batch size')
flags.DEFINE_integer('resume_iter', -1, 'resume iteration')
flags.DEFINE_integer('ensemble', 10, 'number of ensembles')
flags.DEFINE_integer('im_number', 50000, 'number of ensembles')
flags.DEFINE_integer('repeat_scale', 100, 'number of repeat iterations')
flags.DEFINE_float('noise_scale', 0.005, 'amount of noise to output')
flags.DEFINE_integer('idx', 0, 'save index')
flags.DEFINE_integer('nomix', 10, 'number of intervals to stop mixing')
flags.DEFINE_bool('scaled', True, 'whether to scale noise added')
flags.DEFINE_bool('large_model', False, 'whether to use a small or large model')
flags.DEFINE_bool('larger_model', False, 'Whether to use a large model')
flags.DEFINE_bool('wider_model', False, 'Whether to use a large model')
flags.DEFINE_bool('single', False, 'single ')
flags.DEFINE_string('datasource', 'random', 'default or noise or negative or single')
flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or imagenet or imagenetfull')

FLAGS = flags.FLAGS

class InceptionReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._label_storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, ims, labels):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
            self._label_storage.extend(list(labels))
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx+batch_size] = list(ims)
                self._label_storage[self._next_idx:self._next_idx+batch_size] = list(labels)
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = list(ims)[:split_idx]
                self._storage[:batch_size-split_idx] = list(ims)[split_idx:]
                self._label_storage[self._next_idx:] = list(labels)[:split_idx]
                self._label_storage[:batch_size-split_idx] = list(labels)[split_idx:]

        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes):
        ims = []
        labels = []
        for i in idxes:
            ims.append(self._storage[i])
            labels.append(self._label_storage[i])
        return np.array(ims), np.array(labels)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes), idxes

    def set_elms(self, idxes, data, labels):
        for i, ix in enumerate(idxes):
            self._storage[ix] = data[i]
            self._label_storage[ix] = labels[i]


def rescale_im(im):
    return np.clip(im * 256, 0, 255).astype(np.uint8)

def compute_inception(sess, target_vars):
    X_START = target_vars['X_START']
    Y_GT = target_vars['Y_GT']
    X_finals = target_vars['X_finals']
    NOISE_SCALE = target_vars['NOISE_SCALE']
    energy_noise = target_vars['energy_noise']

    size = FLAGS.im_number
    num_steps = size // 1000

    images = []
    test_ims = []


    if FLAGS.dataset == "cifar10":
        test_dataset = Cifar10(full=True, noise=False)
    elif FLAGS.dataset == "imagenet" or FLAGS.dataset == "imagenetfull":
        test_dataset = Imagenet(train=False)

    if FLAGS.dataset != "imagenetfull":
        test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, num_workers=4, shuffle=True, drop_last=False)
    else:
        test_dataloader = TFImagenetLoader('test', FLAGS.batch_size, 0, 1)

    for data_corrupt, data, label_gt in tqdm(test_dataloader):
        data = data.numpy()
        test_ims.extend(list(rescale_im(data)))

        if FLAGS.dataset == "imagenetfull" and len(test_ims) > 60000:
            test_ims = test_ims[:60000]
            break


    # n = min(len(images), len(test_ims))
    print(len(test_ims))
    # fid = get_fid_score(test_ims[:30000], test_ims[-30000:])
    # print("Base FID of score {}".format(fid))

    if FLAGS.dataset == "cifar10":
        classes = 10
    else:
        classes = 1000

    if FLAGS.dataset == "imagenetfull":
        n = 128
    else:
        n = 32

    for j in range(num_steps):
        itr = int(1000 / 500 * FLAGS.repeat_scale)
        data_buffer = InceptionReplayBuffer(1000)
        curr_index = 0

        identity = np.eye(classes)

        for i in tqdm(range(itr)):
            model_index = curr_index % len(X_finals)
            x_final = X_finals[model_index]

            noise_scale = [1]
            if len(data_buffer) < 1000:
                x_init = np.random.uniform(0, 1, (FLAGS.batch_size, n, n, 3))
                label = np.random.randint(0, classes, (FLAGS.batch_size))
                label = identity[label]
                x_new = sess.run([x_final], {X_START:x_init, Y_GT:label, NOISE_SCALE: noise_scale})[0]
                data_buffer.add(x_new, label)
            else:
                (x_init, label), idx = data_buffer.sample(FLAGS.batch_size)
                keep_mask = (np.random.uniform(0, 1, (FLAGS.batch_size)) > 0.99)
                label_keep_mask = (np.random.uniform(0, 1, (FLAGS.batch_size)) > 0.9)
                label_corrupt = np.random.randint(0, classes, (FLAGS.batch_size))
                label_corrupt = identity[label_corrupt]
                x_init_corrupt = np.random.uniform(0, 1, (FLAGS.batch_size, n, n, 3))

                if i < itr - FLAGS.nomix:
                    x_init[keep_mask] = x_init_corrupt[keep_mask]
                    label[label_keep_mask] = label_corrupt[label_keep_mask]
                # else:
                #     noise_scale = [0.7]

                x_new, e_noise = sess.run([x_final, energy_noise], {X_START:x_init, Y_GT:label, NOISE_SCALE: noise_scale})
                data_buffer.set_elms(idx, x_new, label)

                if FLAGS.im_number != 50000:
                    print(np.mean(e_noise), np.std(e_noise))

            curr_index += 1

        ims = np.array(data_buffer._storage[:1000])
        ims = rescale_im(ims)

        images.extend(list(ims))

    saveim = osp.join('/mnt/nfs/yilundu/ebm_code_release/sandbox_cachedir', FLAGS.exp, "test{}.png".format(FLAGS.idx))

    ims = ims[:100]

    if FLAGS.dataset != "imagenetfull":
        im_panel = ims.reshape((10, 10, 32, 32, 3)).transpose((0, 2, 1, 3, 4)).reshape((320, 320, 3))
    else:
        im_panel = ims.reshape((10, 10, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape((1280, 1280, 3))
    imsave(saveim, im_panel)

    print("Saved image!!!!")
    splits = max(1, len(images) // 5000)
    score, std = get_inception_score(images, splits=splits)
    print("Inception score of {} with std of {}".format(score, std))

    # FID score
    # n = min(len(images), len(test_ims))
    fid = get_fid_score(images, test_ims)
    print("FID of score {}".format(fid))




def main(model_list):

    if FLAGS.dataset == "imagenetfull":
        model = ResNet128(num_filters=64)
    elif FLAGS.large_model:
        model = ResNet32Large(num_filters=128)
    elif FLAGS.larger_model:
        model = ResNet32Larger(num_filters=hidden_dim)
    elif FLAGS.wider_model:
        model = ResNet32Wider(num_filters=256, train=False)
    else:
        model = ResNet32(num_filters=128)

    # config = tf.ConfigProto()
    sess = tf.InteractiveSession()

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    weights = []

    for i, model_num in enumerate(model_list):
        weight = model.construct_weights('context_{}'.format(i))
        initialize()
        save_file = osp.join(logdir, 'model_{}'.format(model_num))

        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(i))
        v_map = {(v.name.replace('context_{}'.format(i), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        try:
            saver.restore(sess, save_file)
        except:
            optimistic_remap_restore(sess, save_file, i)
        weights.append(weight)


    if FLAGS.dataset == "imagenetfull":
        X_START = tf.placeholder(shape=(None, 128, 128, 3), dtype = tf.float32)
    else:
        X_START = tf.placeholder(shape=(None, 32, 32, 3), dtype = tf.float32)

    if FLAGS.dataset == "cifar10":
        Y_GT = tf.placeholder(shape=(None, 10), dtype = tf.float32)
    else:
        Y_GT = tf.placeholder(shape=(None, 1000), dtype = tf.float32)

    NOISE_SCALE = tf.placeholder(shape=1, dtype=tf.float32)

    X_finals = []


    # Seperate loops
    for weight in weights:
        X = X_START

        steps = tf.constant(0)
        c = lambda i, x: tf.less(i, FLAGS.num_steps)
        def langevin_step(counter, X):
            scale_rate = 1

            X = X + tf.random_normal(tf.shape(X), mean=0.0, stddev=scale_rate * FLAGS.noise_scale * NOISE_SCALE)

            energy_noise = model.forward(X, weight, label=Y_GT, reuse=True)
            x_grad = tf.gradients(energy_noise, [X])[0]

            if FLAGS.proj_norm != 0.0:
                x_grad = tf.clip_by_value(x_grad, -FLAGS.proj_norm, FLAGS.proj_norm)

            X = X - FLAGS.step_lr * x_grad  * scale_rate
            X = tf.clip_by_value(X, 0, 1)

            counter = counter + 1

            return counter, X

        steps, X = tf.while_loop(c, langevin_step, (steps, X))
        energy_noise = model.forward(X, weight, label=Y_GT, reuse=True)
        X_final = X
        X_finals.append(X_final)

    target_vars = {}
    target_vars['X_START'] = X_START
    target_vars['Y_GT'] = Y_GT
    target_vars['X_finals'] = X_finals
    target_vars['NOISE_SCALE'] = NOISE_SCALE
    target_vars['energy_noise'] = energy_noise

    compute_inception(sess, target_vars)


if __name__ == "__main__":
    # model_list = [117000, 116700]
    model_list = [FLAGS.resume_iter - 300*i for i in range(FLAGS.ensemble)]
    main(model_list)
