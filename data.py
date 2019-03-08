from tensorflow.python.platform import flags
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
import torchvision

FLAGS = flags.FLAGS

# Dataset Options
flags.DEFINE_string('character_path', '/mnt/nfs/yilundu/custom_maml/data/omniglot_resized/Greek', 'path to character to generate with potential function')
flags.DEFINE_string('omniglot_path', '/root/omniglot_resized', 'path to all characters')
flags.DEFINE_string('dsprites_path', '/root/data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', 'path to dsprites characters')
flags.DEFINE_string('cubes_path', '/mnt/nfs/yilundu/pot_kmeans/total_cube.npz', 'path to dsprites characters')
flags.DEFINE_bool('dshape_only', False, 'fix all factors except for shapes')
flags.DEFINE_bool('dpos_only', False, 'fix all factors except for shapes')
flags.DEFINE_bool('dsize_only', False, 'fix all factors except for size of objects')
flags.DEFINE_bool('drot_only', False, 'fix all factors except for rotation of objects')
flags.DEFINE_bool('dsprites_restrict', False, 'fix all factors except for rotation of objects')
flags.DEFINE_string('imagenet_path', '/root/imagenet', 'path to imagenet images')
flags.DEFINE_string('image_path', '/root/miniImagenet/train/', 'path to the imagenet class')
flags.DEFINE_bool('cutout_inside', False, 'whether cutoff should always in image')
flags.DEFINE_float('cutout_prob', 1.0, 'probability of using cutout')
flags.DEFINE_integer('cutout_mask_size', 16, 'size of cutout')
flags.DEFINE_bool('cutout', False, 'whether to add cutout regularizer to training')


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


class OmniglotCharacter(Dataset):

    def __init__(self):
        self.path = FLAGS.character_path
        self.pos_idx = '24'
        self.pos_paths = []
        self.neg_paths = []
        self.pos_images = []

        character_dirs = os.listdir(self.path)

        for cdir in character_dirs:
            char_paths = osp.join(self.path, cdir)
            paths = [osp.join(char_paths, im) for im in os.listdir(char_paths)]

            if self.pos_idx in cdir:
                self.pos_paths.extend(paths)
            else:
                self.neg_paths.extend(paths)

        for pos_path in self.pos_paths:
            self.pos_images.append(imread(pos_path))

        self.pos_images = (255 - np.stack(self.pos_images, axis=0)) / 255.
        self.pos_paths = self.pos_paths * 23
        self.path = self.neg_paths + self.pos_paths

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        label = 1

        if FLAGS.single:
            index = 0

        path = self.pos_paths[index % len(self.pos_paths)]
        im = imread(path)
        im = (255 - im) / 255

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size)
        elif FLAGS.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * np.random.randn(image_size, image_size)
        elif FLAGS.datasource == 'neg':
            path = self.neg_paths[index % len(self.neg_paths)]
            im_corrupt = imread(path)
            im_corrupt = (255 - im_corrupt) / 255


        return im_corrupt, im, label


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
            im_corrupt = np.random.uniform(0, 1,  size=(image_size, image_size, 3))

        return im_corrupt, im, label


class Box2D(Dataset):

    def __init__(self):
        self.minbox = 0.25
        self.maxbox = 0.75

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        im_corrupt = np.random.uniform(0.0, 1.0, (2))
        im = np.random.uniform(self.minbox, self.maxbox, (2))
        label = 0

        return im_corrupt, im, label


class OmniglotFull(Dataset):

    def __init__(self):
        self.path = FLAGS.omniglot_path
        self.paths = []

        alphabet_dirs = os.listdir(self.path)

        for character_dir in alphabet_dirs:
            character_dir = osp.join(self.path, character_dir)
            character_dirs = os.listdir(character_dir)

            for cdir in character_dirs:
                char_paths = osp.join(character_dir, cdir)
                paths = [osp.join(char_paths, im) for im in os.listdir(char_paths)]

                self.paths.extend(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        label = 1

        if FLAGS.single:
            index = 0

        image_size = 28

        path = self.paths[index]
        im = imread(path)
        im = (255 - im) / 255

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size)
        elif FLAGS.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * np.random.randn(image_size, image_size)


        return im_corrupt, im, label

class ImagenetClass(Dataset):

    def __init__(self):
        self.path = FLAGS.image_path
        # use the jellyfish class for training
        self.pos_idx = 'n01910747'
        self.pos_paths = []
        self.neg_paths = []

        image_dirs = os.listdir(self.path)

        for cdir in image_dirs:
            char_paths = osp.join(self.path, cdir)
            paths = [osp.join(char_paths, im) for im in os.listdir(char_paths)]

            if cdir == self.pos_idx:
                self.pos_paths.extend(paths)
            else:
                self.neg_paths.extend(paths)

        self.path = self.pos_paths + self.neg_paths


    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        label = 1
        image_size = 64

        if FLAGS.single:
            index = 0

        path = self.pos_paths[index % len(self.pos_paths)]
        im = imread(path)
        im = imresize(im, (image_size, image_size)) / 255.

        if len(im.shape) != 3:
            print(path)
            return self.__getitem__(index+1)

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'neg':
            path = self.neg_paths[index % len(self.neg_paths)]
            im_corrupt = imread(path)
            im_corrupt = imresize(im_corrupt, (image_size, image_size)) / 255.

            if len(im_corrupt.shape) != 3:
                # print(path)
                return self.__getitem__(index+1)

            im_corrupt = im_corrupt[:, :, :3]

        return im_corrupt, im, label


class Cifar10(Dataset):
    def __init__(self, train=True, full=False, augment=False, noise=True, rescale=1.0):

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
        self.data = CIFAR10("/root/cifar10", transform=transform, train=train, download=True)
        self.test_data = CIFAR10("/root/cifar10", transform=transform, train=False, download=True)
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
            im = im * self.rescale + np.random.uniform(0, self.rescale * 1 / 256., im.shape)

        np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(0.0, self.rescale, (image_size, image_size, 3))

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

        self.data = CIFAR100("/root/cifar100", transform=transform, train=train, download=True)
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
        im = im + np.random.uniform(-1/512, 1/512, im.shape)
        np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(0.0, 1.0, (image_size, image_size, 3))

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
        im = im + np.random.uniform(-1/512, 1/512, im.shape)
        np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(0.0, 1.0, (image_size, image_size, 3))

        return im_corrupt, im, label


class Mnist(Dataset):
    def __init__(self):
        self.data = MNIST("/root/mnist", transform=transforms.ToTensor(), download=True)
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


class Cubes(Dataset):
    def __init__(self, cond_idx=-1):
        dat = np.load(FLAGS.cubes_path)
        self.data = dat['ims']
        self.label = dat['labels']
        self.cond_idx = cond_idx

        if cond_idx != -1:
            mask = np.zeros((1, 24))
            mask[:, cond_idx*6:(cond_idx+1)*6] = 1.

            data_mask = ((self.label * mask).sum(axis=1, keepdims=True) != 0).squeeze()

            self.data = self.data[data_mask]
            self.label = self.label[data_mask]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        cond_idx = self.cond_idx
        im = self.data[index] / 255.

        if cond_idx != -1:
            label = self.label[index, cond_idx*6:(cond_idx+1)*6]
        else:
            label = self.label[index]

        image_size = 64

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * np.random.randn(image_size, image_size, 3)

        return im_corrupt, im, label


class DSprites(Dataset):
    def __init__(self, cond_size=False, cond_shape=False, cond_pos=False, cond_rot=False):
        dat = np.load(FLAGS.dsprites_path)

        if FLAGS.dshape_only:
            l = dat['latents_values']
            mask = (l[:, 4] == 16/31) & (l[:, 5] == 16/31) & (l[:, 2] == 0.5) & (l[:, 3] == 30 * np.pi / 39)
            self.data = np.tile(dat['imgs'][mask], (10000, 1, 1))
            self.label = np.tile(dat['latents_values'][mask], (10000, 1))
            self.label = self.label[:, 1:2]
        elif FLAGS.dpos_only:
            l = dat['latents_values']
            # mask = (l[:, 1] == 1) & (l[:, 2] == 0.5) & (l[:, 3] == 30 * np.pi / 39)
            mask = (l[:, 1] == 1) & (l[:, 3] == 30 * np.pi / 39) & (l[:, 2] == 0.5)
            self.data = np.tile(dat['imgs'][mask], (100, 1, 1))
            self.label = np.tile(dat['latents_values'][mask], (100, 1))
            self.label = self.label[:, 4:] + 0.5
        elif FLAGS.dsize_only:
            l = dat['latents_values']
            # mask = (l[:, 1] == 1) & (l[:, 2] == 0.5) & (l[:, 3] == 30 * np.pi / 39)
            mask = (l[:, 3] == 30 * np.pi / 39) & (l[:, 4] == 16/31) & (l[:, 5] == 16/31) & (l[:, 1] == 1)
            self.data = np.tile(dat['imgs'][mask], (10000, 1, 1))
            self.label = np.tile(dat['latents_values'][mask], (10000, 1))
            self.label = (self.label[:, 2:3])
        elif FLAGS.drot_only:
            l = dat['latents_values']
            mask = (l[:, 2] == 0.5) & (l[:, 4] == 16/31) & (l[:, 5] == 16/31) & (l[:, 1] == 1)
            self.data = np.tile(dat['imgs'][mask], (100, 1, 1))
            self.label = np.tile(dat['latents_values'][mask], (100, 1))
            self.label = (self.label[:, 3:4])
            self.label = np.concatenate([np.cos(self.label), np.sin(self.label)], axis=1)
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
                self.label = np.concatenate([np.cos(self.label), np.sin(self.label)], axis=1)
            else:
                self.label = self.label[:, 1:2]

        self.identity = np.eye(3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        im = self.data[index]
        image_size = 64

        if not (FLAGS.dpos_only or FLAGS.dsize_only) and (not FLAGS.cond_size) and (not FLAGS.cond_pos) and (not FLAGS.cond_rot) and (not FLAGS.drot_only):
            label = self.identity[self.label[index].astype(np.int32)-1].squeeze()
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
                f = pickle.load(open(osp.join(FLAGS.imagenet_path, 'train_data_batch_{}'.format(i)) ,'rb'))
                if i == 1:
                    labels = f['labels']
                    data = f['data']
                else:
                    labels.extend(f['labels'])
                    data = np.vstack((data, f['data']))
        else:
            f = pickle.load(open(osp.join(FLAGS.imagenet_path, 'val_data') ,'rb'))
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
        im = im + np.random.uniform(-1/512, 1/512, im.shape)
        np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(0.0, 1.0, (image_size, image_size, 3))

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
        im = im + np.random.uniform(-1/512, 1/512, im.shape)

        return im, im, label
