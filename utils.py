""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf
import warnings

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
from tensorflow.contrib.framework import sort

FLAGS = flags.FLAGS
flags.DEFINE_integer('spec_iter', 1, 'Number of iterations to normalize spectrum of matrix')
flags.DEFINE_float('spec_norm_val', 1.0, 'Desired norm of matrices')
flags.DEFINE_bool('downsample', False, 'Wheter to do average pool downsampling')
flags.DEFINE_bool('spec_eval', False, 'Set to true to prevent spectral updates')


def get_median(v):
    v = tf.reshape(v, [-1])
    m = tf.shape(v)[0] // 2
    return tf.nn.top_k(v, m)[m - 1]


def set_seed(seed):
    import torch
    import numpy
    import random

    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


def swish(inp):
    return inp * tf.nn.sigmoid(inp)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, ims):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx +
                              batch_size] = list(ims)
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = list(ims)[:split_idx]
                self._storage[:batch_size - split_idx] = list(ims)[split_idx:]
        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes):
        ims = []
        for i in idxes:
            ims.append(self._storage[i])
        return np.array(ims)

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
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)


def get_weight(
        name,
        shape,
        gain=np.sqrt(2),
        use_wscale=False,
        fan_in=None,
        spec_norm=False,
        zero=False,
        fc=False):
    if fan_in is None:
        fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name=name + 'wscale')
        var = tf.get_variable(
            name + 'weight',
            shape=shape,
            initializer=tf.initializers.random_normal()) * wscale
    elif spec_norm:
        if zero:
            var = tf.get_variable(
                shape=shape,
                name=name + 'weight',
                initializer=tf.initializers.random_normal(
                    stddev=1e-10))
            var = spectral_normed_weight(var, name, lower_bound=True, fc=fc)
        else:
            var = tf.get_variable(
                name + 'weight',
                shape=shape,
                initializer=tf.initializers.random_normal())
            var = spectral_normed_weight(var, name, fc=fc)
    else:
        if zero:
            var = tf.get_variable(
                name + 'weight',
                shape=shape,
                initializer=tf.initializers.zero())
        else:
            var = tf.get_variable(
                name + 'weight',
                shape=shape,
                initializer=tf.contrib.layers.xavier_initializer(
                    dtype=tf.float32))

    return var


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x),
                                           axis=[1, 2], keepdims=True) + epsilon)


# helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        def sampler(x): return random.sample(x, nb_samples)
    else:
        def sampler(x): return x
    images = [(i, os.path.join(path, image))
              for i, path in zip(labels, paths)
              for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images


def optimistic_restore(session, save_file, v_prefix=None):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()

    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES) if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            try:
                curr_var = tf.get_variable(saved_var_name)
            except Exception as e:
                print(e)
                continue
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
            else:
                print(var_name)
                print(var_shape, saved_shapes[saved_var_name])

    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def optimistic_remap_restore(session, save_file, v_prefix):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()

    vars_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope='context_{}'.format(v_prefix))
    var_names = sorted([(var.name.split(':')[0], var) for var in vars_list if (
        (var.name.split(':')[0]).replace('context_{}'.format(v_prefix), 'context_0') in saved_shapes)])
    restore_vars = []

    v_map = {}
    with tf.variable_scope('', reuse=True):
        for saved_var_name, curr_var in var_names:
            var_shape = curr_var.get_shape().as_list()
            saved_var_name = saved_var_name.replace(
                'context_{}'.format(v_prefix), 'context_0')
            if var_shape == saved_shapes[saved_var_name]:
                v_map[saved_var_name] = curr_var
            else:
                print(saved_var_name)
                print(var_shape, saved_shapes[saved_var_name])

    saver = tf.train.Saver(v_map)
    saver.restore(session, save_file)


def remap_restore(session, save_file, i):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            try:
                curr_var = tf.get_variable(saved_var_name)
            except Exception as e:
                print(e)
                continue
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    print(restore_vars)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


# Network weight initializers
def init_conv_weight(
        weights,
        scope,
        k,
        c_in,
        c_out,
        spec_norm=True,
        zero=False,
        scale=1.0,
        classes=1):

    if spec_norm:
        spec_norm = FLAGS.spec_norm

    conv_weights = {}
    with tf.variable_scope(scope):
        if zero:
            conv_weights['c'] = get_weight(
                'c', [k, k, c_in, c_out], spec_norm=spec_norm, zero=True)
        else:
            conv_weights['c'] = get_weight(
                'c', [k, k, c_in, c_out], spec_norm=spec_norm)

        conv_weights['b'] = tf.get_variable(
            shape=[c_out], name='b', initializer=tf.initializers.zeros())

        if classes != 1:
            conv_weights['g'] = tf.get_variable(
                shape=[
                    classes,
                    c_out],
                name='g',
                initializer=tf.initializers.ones())
            conv_weights['gb'] = tf.get_variable(
                shape=[
                    classes,
                    c_in],
                name='gb',
                initializer=tf.initializers.zeros())
        else:
            conv_weights['g'] = tf.get_variable(
                shape=[c_out], name='g', initializer=tf.initializers.ones())
            conv_weights['gb'] = tf.get_variable(
                shape=[c_in], name='gb', initializer=tf.initializers.zeros())

        conv_weights['cb'] = tf.get_variable(
            shape=[c_in], name='cb', initializer=tf.initializers.zeros())

    weights[scope] = conv_weights


def init_convt_weight(
        weights,
        scope,
        k,
        c_in,
        c_out,
        spec_norm=True,
        zero=False,
        scale=1.0,
        classes=1):

    if spec_norm:
        spec_norm = FLAGS.spec_norm

    conv_weights = {}
    with tf.variable_scope(scope):
        if zero:
            conv_weights['c'] = get_weight(
                'c', [k, k, c_in, c_out], spec_norm=spec_norm, zero=True)
        else:
            conv_weights['c'] = get_weight(
                'c', [k, k, c_in, c_out], spec_norm=spec_norm)

        conv_weights['b'] = tf.get_variable(
            shape=[c_in], name='b', initializer=tf.initializers.zeros())

        if classes != 1:
            conv_weights['g'] = tf.get_variable(
                shape=[
                    classes,
                    c_in],
                name='g',
                initializer=tf.initializers.ones())
            conv_weights['gb'] = tf.get_variable(
                shape=[
                    classes,
                    c_out],
                name='gb',
                initializer=tf.initializers.zeros())
        else:
            conv_weights['g'] = tf.get_variable(
                shape=[c_in], name='g', initializer=tf.initializers.ones())
            conv_weights['gb'] = tf.get_variable(
                shape=[c_out], name='gb', initializer=tf.initializers.zeros())

        conv_weights['cb'] = tf.get_variable(
            shape=[c_in], name='cb', initializer=tf.initializers.zeros())

    weights[scope] = conv_weights


def init_attention_weight(
        weights,
        scope,
        c_in,
        k,
        trainable_gamma=True,
        spec_norm=True):

    if spec_norm:
        spec_norm = FLAGS.spec_norm

    atten_weights = {}
    with tf.variable_scope(scope):
        atten_weights['q'] = get_weight(
            'atten_q', [1, 1, c_in, k], spec_norm=spec_norm)
        atten_weights['q_b'] = tf.get_variable(
            shape=[k], name='atten_q_b1', initializer=tf.initializers.zeros())
        atten_weights['k'] = get_weight(
            'atten_k', [1, 1, c_in, k], spec_norm=spec_norm)
        atten_weights['k_b'] = tf.get_variable(
            shape=[k], name='atten_k_b1', initializer=tf.initializers.zeros())
        atten_weights['v'] = get_weight(
            'atten_v', [1, 1, c_in, c_in], spec_norm=spec_norm)
        atten_weights['v_b'] = tf.get_variable(
            shape=[c_in], name='atten_v_b1', initializer=tf.initializers.zeros())
        atten_weights['gamma'] = tf.get_variable(
            shape=[1], name='gamma', initializer=tf.initializers.zeros())

    weights[scope] = atten_weights


def init_fc_weight(weights, scope, c_in, c_out, spec_norm=True):
    fc_weights = {}

    if spec_norm:
        spec_norm = FLAGS.spec_norm

    with tf.variable_scope(scope):
        fc_weights['w'] = get_weight(
            'w', [c_in, c_out], spec_norm=spec_norm, fc=True)
        fc_weights['b'] = tf.get_variable(
            shape=[c_out], name='b', initializer=tf.initializers.zeros())

    weights[scope] = fc_weights


def init_res_weight(
        weights,
        scope,
        k,
        c_in,
        c_out,
        hidden_dim=None,
        spec_norm=True,
        res_scale=1.0,
        classes=1):

    if not hidden_dim:
        hidden_dim = c_in

    if spec_norm:
        spec_norm = FLAGS.spec_norm

    init_conv_weight(
        weights,
        scope +
        '_res_c1',
        k,
        c_in,
        c_out,
        spec_norm=spec_norm,
        scale=res_scale,
        classes=classes)
    init_conv_weight(
        weights,
        scope + '_res_c2',
        k,
        c_out,
        c_out,
        spec_norm=spec_norm,
        zero=True,
        scale=res_scale,
        classes=classes)

    if c_in != c_out:
        init_conv_weight(
            weights,
            scope +
            '_res_adaptive',
            k,
            c_in,
            c_out,
            spec_norm=spec_norm,
            scale=res_scale,
            classes=classes)

# Network forward helpers


def smart_conv_block(inp, weights, reuse, scope, use_stride=True, **kwargs):
    weights = weights[scope]
    return conv_block(
        inp,
        weights['c'],
        weights['b'],
        reuse,
        scope,
        scale=weights['g'],
        bias=weights['gb'],
        class_bias=weights['cb'],
        use_stride=use_stride,
        **kwargs)


def smart_convt_block(
        inp,
        weights,
        reuse,
        scope,
        output_dim,
        upsample=True,
        label=None):
    weights = weights[scope]

    cweight = weights['c']
    bweight = weights['b']
    scale = weights['g']
    bias = weights['gb']
    class_bias = weights['cb']

    if upsample:
        stride = [1, 2, 2, 1]
    else:
        stride = [1, 1, 1, 1]

    if label is not None:
        bias_batch = tf.matmul(label, bias)
        batch = tf.shape(bias_batch)[0]
        dim = tf.shape(bias_batch)[1]
        bias = tf.reshape(bias_batch, (batch, 1, 1, dim))

        inp = inp + bias

    shape = cweight.get_shape()
    conv_output = tf.nn.conv2d_transpose(inp,
                                         cweight,
                                         [tf.shape(inp)[0],
                                          output_dim,
                                          output_dim,
                                          cweight.get_shape().as_list()[-2]],
                                         stride,
                                         'SAME')

    if label is not None:
        scale_batch = tf.matmul(label, scale) + class_bias
        batch = tf.shape(scale_batch)[0]
        dim = tf.shape(scale_batch)[1]
        scale = tf.reshape(scale_batch, (batch, 1, 1, dim))

        conv_output = conv_output * scale

    conv_output = tf.nn.leaky_relu(conv_output)

    return conv_output


def smart_res_block(
        inp,
        weights,
        reuse,
        scope,
        downsample=True,
        adaptive=True,
        stop_batch=False,
        upsample=False,
        label=None,
        act=tf.nn.leaky_relu,
        dropout=False,
        train=False,
        **kwargs):
    gn1 = weights[scope + '_res_c1']
    gn2 = weights[scope + '_res_c2']
    c1 = smart_conv_block(
        inp,
        weights,
        reuse,
        scope + '_res_c1',
        use_stride=False,
        activation=None,
        extra_bias=True,
        label=label,
        **kwargs)

    if dropout:
        c1 = tf.layers.dropout(c1, rate=0.5, training=train)

    c1 = act(c1)
    c2 = smart_conv_block(
        c1,
        weights,
        reuse,
        scope + '_res_c2',
        use_stride=False,
        activation=None,
        use_scale=True,
        extra_bias=True,
        label=label,
        **kwargs)

    if adaptive:
        c_bypass = smart_conv_block(
            inp,
            weights,
            reuse,
            scope +
            '_res_adaptive',
            use_stride=False,
            activation=None,
            **kwargs)
    else:
        c_bypass = inp

    res = c2 + c_bypass

    if upsample:
        res_shape = tf.shape(res)
        res_shape_list = res.get_shape()
        res = tf.image.resize_nearest_neighbor(
            res, [2 * res_shape_list[1], 2 * res_shape_list[2]])
    elif downsample:
        res = tf.nn.avg_pool(res, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

    res = act(res)

    return res


def smart_res_block_optim(inp, weights, reuse, scope, **kwargs):
    c1 = smart_conv_block(
        inp,
        weights,
        reuse,
        scope + '_res_c1',
        use_stride=False,
        activation=None,
        **kwargs)
    c1 = tf.nn.leaky_relu(c1)
    c2 = smart_conv_block(
        c1,
        weights,
        reuse,
        scope + '_res_c2',
        use_stride=False,
        activation=None,
        **kwargs)

    inp = tf.nn.avg_pool(inp, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
    c_bypass = smart_conv_block(
        inp,
        weights,
        reuse,
        scope +
        '_res_adaptive',
        use_stride=False,
        activation=None,
        **kwargs)
    c2 = tf.nn.avg_pool(c2, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

    res = c2 + c_bypass

    return c2


def groupsort(k=4):
    def sortact(inp):
        old_shape = tf.shape(inp)
        inp = sort(tf.reshape(inp, (-1, 4)))
        inp = tf.reshape(inp, old_shape)
        return inp
    return sortact


def smart_atten_block(inp, weights, reuse, scope, **kwargs):
    w = weights[scope]
    return attention(
        inp,
        w['q'],
        w['q_b'],
        w['k'],
        w['k_b'],
        w['v'],
        w['v_b'],
        w['gamma'],
        reuse,
        scope,
        **kwargs)


def smart_fc_block(inp, weights, reuse, scope, use_bias=True):
    weights = weights[scope]
    output = tf.matmul(inp, weights['w'])

    if use_bias:
        output = output + weights['b']

    return output


# Network helpers
def conv_block(
        inp,
        cweight,
        bweight,
        reuse,
        scope,
        use_stride=True,
        activation=tf.nn.leaky_relu,
        pn=False,
        bn=False,
        gn=False,
        ln=False,
        scale=None,
        bias=None,
        class_bias=None,
        use_bias=False,
        downsample=False,
        stop_batch=False,
        use_scale=False,
        extra_bias=False,
        average=False,
        label=None):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
    _, h, w, _ = inp.get_shape()

    if FLAGS.downsample:
        stride = no_stride

    if not FLAGS.use_bias and not use_bias:
        bweight = 0

    if extra_bias:
        if label is not None:
            if len(bias.get_shape()) == 1:
                bias = tf.reshape(bias, (1, -1))
            bias_batch = tf.matmul(label, bias)
            batch = tf.shape(bias_batch)[0]
            dim = tf.shape(bias_batch)[1]
            bias = tf.reshape(bias_batch, (batch, 1, 1, dim))

        inp = inp + bias

    if not use_stride:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME')
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME')

    if use_scale:
        if label is not None:
            if len(scale.get_shape()) == 1:
                scale = tf.reshape(scale, (1, -1))
            scale_batch = tf.matmul(label, scale) + class_bias
            batch = tf.shape(scale_batch)[0]
            dim = tf.shape(scale_batch)[1]
            scale = tf.reshape(scale_batch, (batch, 1, 1, dim))

        conv_output = conv_output * scale

    if use_bias:
        conv_output = conv_output + bweight

    if activation is not None:
        conv_output = activation(conv_output)

    if bn:
        conv_output = batch_norm(conv_output, scale, bias)
    if pn:
        conv_output = pixel_norm(conv_output)
    if gn:
        conv_output = group_norm(
            conv_output, scale, bias, stop_batch=stop_batch)
    if ln:
        conv_output = layer_norm(conv_output, scale, bias)

    if FLAGS.downsample and use_stride:
        conv_output = tf.layers.average_pooling2d(conv_output, (2, 2), 2)

    return conv_output


def conv_block_1d(
        inp,
        cweight,
        bweight,
        reuse,
        scope,
        activation=tf.nn.leaky_relu):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride = 1

    conv_output = tf.nn.conv1d(inp, cweight, stride, 'SAME') + bweight

    if activation is not None:
        conv_output = activation(conv_output)

    return conv_output


def conv_block_3d(
        inp,
        cweight,
        bweight,
        reuse,
        scope,
        use_stride=True,
        activation=tf.nn.leaky_relu,
        pn=False,
        bn=False,
        gn=False,
        ln=False,
        scale=None,
        bias=None,
        use_bias=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1, 1, 2, 2, 1], [1, 1, 1, 1, 1]
    _, d, h, w, _ = inp.get_shape()

    if not FLAGS.use_bias and not use_bias:
        bweight = 0

    if not use_stride:
        conv_output = tf.nn.conv3d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv3d(inp, cweight, stride, 'SAME') + bweight

    if activation is not None:
        conv_output = activation(conv_output, alpha=0.1)

    if bn:
        conv_output = batch_norm(conv_output, scale, bias)
    if pn:
        conv_output = pixel_norm(conv_output)
    if gn:
        conv_output = group_norm(conv_output, scale, bias)
    if ln:
        conv_output = layer_norm(conv_output, scale, bias)

    if FLAGS.downsample and use_stride:
        conv_output = tf.layers.average_pooling2d(conv_output, (2, 2), 2)

    return conv_output


def group_norm(inp, scale, bias, g=32, eps=1e-6, stop_batch=False):
    """Applies group normalization assuming nhwc format"""
    n, h, w, c = inp.shape
    inp = tf.reshape(inp, (tf.shape(inp)[0], h, w, c // g, g))

    mean, var = tf.nn.moments(inp, [1, 2, 4], keep_dims=True)
    gain = tf.rsqrt(var + eps)

    # if stop_batch:
    #     gain = tf.stop_gradient(gain)

    output = gain * (inp - mean)
    output = tf.reshape(output, (tf.shape(inp)[0], h, w, c))

    if scale is not None:
        output = output * scale

    if bias is not None:
        output = output + bias

    return output


def layer_norm(inp, scale, bias, eps=1e-6):
    """Applies group normalization assuming nhwc format"""
    n, h, w, c = inp.shape

    mean, var = tf.nn.moments(inp, [1, 2, 3], keep_dims=True)
    gain = tf.rsqrt(var + eps)
    output = gain * (inp - mean)

    if scale is not None:
        output = output * scale

    if bias is not None:
        output = output + bias

    return output


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    y_shapes = tf.shape(y)

    return tf.concat(
        [x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]]) / 10.], 3)


def attention(
        inp,
        q,
        q_b,
        k,
        k_b,
        v,
        v_b,
        gamma,
        reuse,
        scope,
        stop_at_grad=False,
        seperate=False,
        scale=False,
        train=False,
        dropout=0.0):
    conv_q = conv_block(
        inp,
        q,
        q_b,
        reuse=reuse,
        scope=scope,
        use_stride=False,
        activation=None,
        use_bias=True,
        pn=False,
        bn=False,
        gn=False)
    conv_k = conv_block(
        inp,
        k,
        k_b,
        reuse=reuse,
        scope=scope,
        use_stride=False,
        activation=None,
        use_bias=True,
        pn=False,
        bn=False,
        gn=False)

    conv_v = conv_block(
        inp,
        v,
        v_b,
        reuse=reuse,
        scope=scope,
        use_stride=False,
        pn=False,
        bn=False,
        gn=False)

    c_num = float(conv_q.get_shape().as_list()[-1])
    s = tf.matmul(hw_flatten(conv_q), hw_flatten(conv_k), transpose_b=True)

    if scale:
        s = s / (c_num) ** 0.5

    if train:
        s = tf.nn.dropout(s, 0.9)

    beta = tf.nn.softmax(s, axis=-1)
    o = tf.matmul(beta, hw_flatten(conv_v))
    o = tf.reshape(o, shape=tf.shape(inp))
    inp = inp + gamma * o

    if not seperate:
        return inp
    else:
        return gamma * o


def attention_2d(
        inp,
        q,
        q_b,
        k,
        k_b,
        v,
        v_b,
        reuse,
        scope,
        stop_at_grad=False,
        seperate=False,
        scale=False):
    inp_shape = tf.shape(inp)
    inp_compact = tf.reshape(
        inp,
        (inp_shape[0] *
         FLAGS.input_objects *
         inp_shape[1],
         inp.shape[3]))
    f_q = tf.matmul(inp_compact, q) + q_b
    f_k = tf.matmul(inp_compact, k) + k_b
    f_v = tf.nn.leaky_relu(tf.matmul(inp_compact, v) + v_b)

    f_q = tf.reshape(f_q,
                     (inp_shape[0],
                      inp_shape[1],
                         inp_shape[2],
                         tf.shape(f_q)[-1]))
    f_k = tf.reshape(f_k,
                     (inp_shape[0],
                      inp_shape[1],
                         inp_shape[2],
                         tf.shape(f_k)[-1]))
    f_v = tf.reshape(
        f_v,
        (inp_shape[0],
         inp_shape[1],
         inp_shape[2],
         inp_shape[3]))

    s = tf.matmul(f_k, f_q, transpose_b=True)
    c_num = (32**0.5)

    if scale:
        s = s / c_num

    beta = tf.nn.softmax(s, axis=-1)

    o = tf.reshape(tf.matmul(beta, f_v), inp_shape) + inp

    return o


def hw_flatten(x):
    shape = tf.shape(x)
    return tf.reshape(x, [tf.shape(x)[0], -1, shape[-1]])


def batch_norm(inp, scale, bias, eps=0.01):
    mean, var = tf.nn.moments(inp, [0])
    output = tf.nn.batch_normalization(inp, mean, var, bias, scale, eps)
    return output


def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(
            inp,
            activation_fn=activation,
            reuse=reuse,
            scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(
            inp,
            activation_fn=activation,
            reuse=reuse,
            scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

# Loss functions


def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred - label))


NO_OPS = 'NO_OPS'


def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(w, name, lower_bound=False, iteration=1, fc=False):
    if fc:
        iteration = 2

    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    iteration = FLAGS.spec_iter
    sigma_new = FLAGS.spec_norm_val

    u = tf.get_variable(name + "_u",
                        [1,
                         w_shape[-1]],
                        initializer=tf.random_normal_initializer(),
                        trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    if FLAGS.spec_eval:
        dep = []
    else:
        dep = [u.assign(u_hat)]

    with tf.control_dependencies(dep):
        if lower_bound:
            sigma = sigma + 1e-6
            w_norm = w / sigma * tf.minimum(sigma, 1) * sigma_new
        else:
            w_norm = w / sigma * sigma_new

        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, v in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over
                # below.
                grads.append(expanded_g)
            else:
                print(g, v)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
