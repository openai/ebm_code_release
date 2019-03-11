from tensorflow.python.platform import flags
from tensorflow.contrib.data.python.ops import batching, threadpool
import tensorflow as tf
import json
from torch.utils.data import Dataset
import pickle
import os.path as osp
import os
import numpy as np
import time
from scipy.misc import imread, imresize
from skimage.color import rgb2grey
from torchvision.datasets import CIFAR10, MNIST, SVHN, CIFAR100, ImageFolder
from torchvision import transforms
from imagenet_preprocessing import ImagenetPreprocessor
import torch
import torchvision

FLAGS = flags.FLAGS

# Dataset Options
flags.DEFINE_string('dsprites_path',
    '/root/data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
    'path to dsprites characters')
flags.DEFINE_string('imagenet_datadir',  '/root/imagenet_big', 'whether cutoff should always in image')
flags.DEFINE_bool('dshape_only', False, 'fix all factors except for shapes')
flags.DEFINE_bool('dpos_only', False, 'fix all factors except for positions of shapes')
flags.DEFINE_bool('dsize_only', False,'fix all factors except for size of objects')
flags.DEFINE_bool('drot_only', False, 'fix all factors except for rotation of objects')
flags.DEFINE_bool('dsprites_restrict', False, 'fix all factors except for rotation of objects')
flags.DEFINE_string('imagenet_path', '/root/imagenet', 'path to imagenet images')


# Data augmentation options
flags.DEFINE_bool('cutout_inside', False,'whether cutoff should always in image')
flags.DEFINE_float('cutout_prob', 1.0, 'probability of using cutout')
flags.DEFINE_integer('cutout_mask_size', 16, 'size of cutout')
flags.DEFINE_bool('cutout', False,'whether to add cutout regularizer to data')


def cutout(mask_color=(0, 0, 0)):
    mask_size_half = FLAGS.cutout_mask_size // 2
    offset = 1 if FLAGS.cutout_mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > FLAGS.cutout_prob:
            return image

        h, w = image.shape[:2]

        if FLAGS.cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + FLAGS.cutout_mask_size
        ymax = ymin + FLAGS.cutout_mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[:, ymin:ymax, xmin:xmax] = np.array(mask_color)[:, None, None]
        return image

    return _cutout


class TFImagenetLoader(Dataset):

    def __init__(self, split, batchsize, idx, num_workers, rescale=1):
        IMAGENET_NUM_TRAIN_IMAGES = 1281167
        IMAGENET_NUM_VAL_IMAGES = 50000

        self.rescale = rescale

        if split == "train":
            im_length = IMAGENET_NUM_TRAIN_IMAGES
            records_to_skip = im_length * idx // num_workers
            records_to_read = im_length * (idx + 1) // num_workers - records_to_skip
        else:
            im_length = IMAGENET_NUM_VAL_IMAGES

        self.curr_sample = 0

        index_path = osp.join(FLAGS.imagenet_datadir, 'index.json')
        with open(index_path) as f:
            metadata = json.load(f)
            counts = metadata['record_counts']

        if split == 'train':
            file_names = list(sorted([x for x in counts.keys() if x.startswith('train')]))

            result_records_to_skip = None
            files = []
            for filename in file_names:
                records_in_file = counts[filename]
                if records_to_skip >= records_in_file:
                    records_to_skip -= records_in_file
                    continue
                elif records_to_read > 0:
                    if result_records_to_skip is None:
                        # Record the number to skip in the first file
                        result_records_to_skip = records_to_skip
                    files.append(filename)
                    records_to_read -= (records_in_file - records_to_skip)
                    records_to_skip = 0
                else:
                    break
        else:
            files = list(sorted([x for x in counts.keys() if x.startswith('validation')]))

        files = [osp.join(FLAGS.imagenet_datadir, x) for x in files]
        preprocess_function = ImagenetPreprocessor(128, dtype=tf.float32, train=False).parse_and_preprocess

        ds = tf.data.TFRecordDataset.from_generator(lambda: files, output_types=tf.string)
        ds = ds.apply(tf.data.TFRecordDataset)
        ds = ds.take(im_length)
        ds = ds.prefetch(buffer_size=FLAGS.batch_size)
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
        ds = ds.apply(batching.map_and_batch(map_func=preprocess_function, batch_size=FLAGS.batch_size, num_parallel_batches=4))
        ds = ds.prefetch(buffer_size=2)

        ds_iterator = ds.make_initializable_iterator()
        labels, images = ds_iterator.get_next()
        self.images = tf.clip_by_value(images / 256 + tf.random_uniform(tf.shape(images), 0, 1. / 256), 0.0, 1.0)
        self.labels = labels

        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)
        sess.run(ds_iterator.initializer)

        self.im_length = im_length // batchsize

        self.sess = sess

    def __next__(self):
        self.curr_sample += 1

        sess = self.sess

        im_corrupt = np.random.uniform(0, self.rescale, size=(FLAGS.batch_size, 128, 128, 3))
        label, im = sess.run([self.labels, self.images])
        im = im * self.rescale
        label = np.eye(1000)[label.squeeze() - 1]
        im, im_corrupt, label = torch.from_numpy(im), torch.from_numpy(im_corrupt), torch.from_numpy(label)
        return im_corrupt, im, label

    def __iter__(self):
        return self

class CelebA(Dataset):

    def __init__(self):
        self.path = "/root/data/img_align_celeba"
        self.ims = os.listdir(self.path)
        self.ims = [osp.join(self.path, im) for im in self.ims]

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, index):
        label = 1

        if FLAGS.single:
            index = 0

        path = self.ims[index]
        im = imread(path)
        im = imresize(im, (32, 32))
        image_size = 32
        im = im / 255.

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(
                0, 1, size=(image_size, image_size, 3))

        return im_corrupt, im, label


class Cifar10(Dataset):
    def __init__(
            self,
            train=True,
            full=False,
            augment=False,
            noise=True,
            rescale=1.0):

        if augment:
            transform_list = [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ]

            if FLAGS.cutout:
                transform_list.append(cutout())

            transform = transforms.Compose(transform_list)
        else:
            transform = transforms.ToTensor()

        self.full = full
        self.data = CIFAR10(
            "/root/cifar10",
            transform=transform,
            train=train,
            download=True)
        self.test_data = CIFAR10(
            "/root/cifar10",
            transform=transform,
            train=False,
            download=True)
        self.one_hot_map = np.eye(10)
        self.noise = noise
        self.rescale = rescale

    def __len__(self):

        if self.full:
            return len(self.data) + len(self.test_data)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if not FLAGS.single:
            if self.full:
                if index >= len(self.data):
                    im, label = self.test_data[index - len(self.data)]
                else:
                    im, label = self.data[index]
            else:
                im, label = self.data[index]
        else:
            im, label = self.data[0]

        im = np.transpose(im, (1, 2, 0)).numpy()
        image_size = 32
        label = self.one_hot_map[label]

        im = im * 255 / 256

        if self.noise:
            im = im * self.rescale + \
                np.random.uniform(0, self.rescale * 1 / 256., im.shape)

        np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(
                0.0, self.rescale, (image_size, image_size, 3))

        return im_corrupt, im, label


class Cifar100(Dataset):
    def __init__(self, train=True, augment=False):

        if augment:
            transform_list = [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ]

            if FLAGS.cutout:
                transform_list.append(cutout())

            transform = transforms.Compose(transform_list)
        else:
            transform = transforms.ToTensor()

        self.data = CIFAR100(
            "/root/cifar100",
            transform=transform,
            train=train,
            download=True)
        self.one_hot_map = np.eye(100)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not FLAGS.single:
            im, label = self.data[index]
        else:
            im, label = self.data[0]

        im = np.transpose(im, (1, 2, 0)).numpy()
        image_size = 32
        label = self.one_hot_map[label]
        im = im + np.random.uniform(-1 / 512, 1 / 512, im.shape)
        np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(
                0.0, 1.0, (image_size, image_size, 3))

        return im_corrupt, im, label


class Svhn(Dataset):
    def __init__(self, train=True, augment=False):

        transform = transforms.ToTensor()

        self.data = SVHN("/root/svhn", transform=transform, download=True)
        self.one_hot_map = np.eye(10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not FLAGS.single:
            im, label = self.data[index]
        else:
            em, label = self.data[0]

        im = np.transpose(im, (1, 2, 0)).numpy()
        image_size = 32
        label = self.one_hot_map[label]
        im = im + np.random.uniform(-1 / 512, 1 / 512, im.shape)
        np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(
                0.0, 1.0, (image_size, image_size, 3))

        return im_corrupt, im, label


class Mnist(Dataset):
    def __init__(self):
        self.data = MNIST(
            "/root/mnist",
            transform=transforms.ToTensor(),
            download=True)
        self.labels = np.eye(10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        im, label = self.data[index]
        label = self.labels[label]
        im = im.squeeze()
        image_size = 28

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size)
        elif FLAGS.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * np.random.randn(image_size, image_size)

        return im_corrupt, im, label


class DSprites(Dataset):
    def __init__(
            self,
            cond_size=False,
            cond_shape=False,
            cond_pos=False,
            cond_rot=False):
        dat = np.load(FLAGS.dsprites_path)

        if FLAGS.dshape_only:
            l = dat['latents_values']
            mask = (l[:, 4] == 16 / 31) & (l[:, 5] == 16 /
                                           31) & (l[:, 2] == 0.5) & (l[:, 3] == 30 * np.pi / 39)
            self.data = np.tile(dat['imgs'][mask], (10000, 1, 1))
            self.label = np.tile(dat['latents_values'][mask], (10000, 1))
            self.label = self.label[:, 1:2]
        elif FLAGS.dpos_only:
            l = dat['latents_values']
            # mask = (l[:, 1] == 1) & (l[:, 2] == 0.5) & (l[:, 3] == 30 * np.pi / 39)
            mask = (l[:, 1] == 1) & (
                l[:, 3] == 30 * np.pi / 39) & (l[:, 2] == 0.5)
            self.data = np.tile(dat['imgs'][mask], (100, 1, 1))
            self.label = np.tile(dat['latents_values'][mask], (100, 1))
            self.label = self.label[:, 4:] + 0.5
        elif FLAGS.dsize_only:
            l = dat['latents_values']
            # mask = (l[:, 1] == 1) & (l[:, 2] == 0.5) & (l[:, 3] == 30 * np.pi / 39)
            mask = (l[:, 3] == 30 * np.pi / 39) & (l[:, 4] == 16 /
                                                   31) & (l[:, 5] == 16 / 31) & (l[:, 1] == 1)
            self.data = np.tile(dat['imgs'][mask], (10000, 1, 1))
            self.label = np.tile(dat['latents_values'][mask], (10000, 1))
            self.label = (self.label[:, 2:3])
        elif FLAGS.drot_only:
            l = dat['latents_values']
            mask = (l[:, 2] == 0.5) & (l[:, 4] == 16 /
                                       31) & (l[:, 5] == 16 / 31) & (l[:, 1] == 1)
            self.data = np.tile(dat['imgs'][mask], (100, 1, 1))
            self.label = np.tile(dat['latents_values'][mask], (100, 1))
            self.label = (self.label[:, 3:4])
            self.label = np.concatenate(
                [np.cos(self.label), np.sin(self.label)], axis=1)
        elif FLAGS.dsprites_restrict:
            l = dat['latents_values']
            mask = (l[:, 1] == 1) & (l[:, 3] == 0 * np.pi / 39)

            self.data = dat['imgs'][mask]
            self.label = dat['latents_values'][mask]
        else:
            self.data = dat['imgs']
            self.label = dat['latents_values']

            if cond_size:
                self.label = self.label[:, 2:3]
            elif cond_shape:
                self.label = self.label[:, 1:2]
            elif cond_pos:
                self.label = self.label[:, 4:]
            elif cond_rot:
                self.label = self.label[:, 3:4]
                self.label = np.concatenate(
                    [np.cos(self.label), np.sin(self.label)], axis=1)
            else:
                self.label = self.label[:, 1:2]

        self.identity = np.eye(3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        im = self.data[index]
        image_size = 64

        if not (
            FLAGS.dpos_only or FLAGS.dsize_only) and (
            not FLAGS.cond_size) and (
            not FLAGS.cond_pos) and (
                not FLAGS.cond_rot) and (
                    not FLAGS.drot_only):
            label = self.identity[self.label[index].astype(
                np.int32) - 1].squeeze()
        else:
            label = self.label[index]

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size)
        elif FLAGS.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * np.random.randn(image_size, image_size)

        return im_corrupt, im, label


class Imagenet(Dataset):
    def __init__(self, train=True, augment=False):

        if train:
            for i in range(1, 11):
                f = pickle.load(
                    open(
                        osp.join(
                            FLAGS.imagenet_path,
                            'train_data_batch_{}'.format(i)),
                        'rb'))
                if i == 1:
                    labels = f['labels']
                    data = f['data']
                else:
                    labels.extend(f['labels'])
                    data = np.vstack((data, f['data']))
        else:
            f = pickle.load(
                open(
                    osp.join(
                        FLAGS.imagenet_path,
                        'val_data'),
                    'rb'))
            labels = f['labels']
            data = f['data']

        self.labels = labels
        self.data = data
        self.one_hot_map = np.eye(1000)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if not FLAGS.single:
            im, label = self.data[index], self.labels[index]
        else:
            im, label = self.data[0], self.labels[0]

        label -= 1

        im = im.reshape((3, 32, 32)) / 255
        im = im.transpose((1, 2, 0))
        image_size = 32
        label = self.one_hot_map[label]
        im = im + np.random.uniform(-1 / 512, 1 / 512, im.shape)
        np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(
                0.0, 1.0, (image_size, image_size, 3))

        return im_corrupt, im, label


class Textures(Dataset):
    def __init__(self, train=True, augment=False):
        self.dataset = ImageFolder("/mnt/nfs/yilundu/data/dtd/images")

    def __len__(self):
        return 2 * len(self.dataset)

    def __getitem__(self, index):
        idx = index % (len(self.dataset))
        im, label = self.dataset[idx]

        im = np.array(im)[:32, :32] / 255
        im = im + np.random.uniform(-1 / 512, 1 / 512, im.shape)

        return im, im, label
