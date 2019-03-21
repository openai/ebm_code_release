# EBM

Code for Implicit Generation and Generalization in Energy Based Models. Blog post can be found [here]() and website with pretrained models can be found [here](https://sites.google.com/view/igebm/home)

## Requirements

To install the prerequisites for the project run 
```
pip install -r requirements.txt
```

Download all saved models in the folder cachedir

## Download Datasets

For MNIST and CIFAR-10 datasets, code will directly download the data.

For ImageNet 128x128 dataset, download the TFRecords of the Imagenet dataset by running in an 
ImageNet folder

```
for i in $(seq -f "%05g" 0 1023)
do
  wget https://storage.googleapis.com/ebm_demo/data/imagenet/train-$i-of-01024
done

for i in $(seq -f "%05g" 0 127)
do
  wget https://storage.googleapis.com/ebm_demo/data/imagenet/validation-$i-of-00128
done

wget https://storage.googleapis.com/ebm_demo/data/imagenet/index.json
```

For Imagenet 32x32 dataset, download the Imagenet 32x32 dataset and unzip by running

```
wget https://storage.googleapis.com/ebm_demo/data/imagenet32/Imagenet32_train.zip
wget https://storage.googleapis.com/ebm_demo/data/imagenet32/Imagenet32_val.zip
```

For dSprites dataset, download the dataset by running

```
wget https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true
```

## Training

To train on different datasets:

For CIFAR-10 Unconditional

```
python train.py --exp=cifar10_uncond --dataset=cifar10 --num_steps=60 --batch_size=128 --step_lr=10.0 --proj_norm=0.01 --zero_kl --replay_batch --large_network
```

For CIFAR-10 Conditional

```
python train.py --exp=cifar10_cond --dataset=cifar10 --num_steps=60 --batch_size=128 --step_lr=10.0 --proj_norm=0.01 --zero_kl --replay_batch --cclass
```

For ImageNet 32x32 Conditional

```
python train.py --exp=imagenet_cond --num_steps=60  --wider_model --batch_size=32 step_lr=10.0 --proj_norm=0.01 --replay_batch --cclass --zero_kl --dataset=imagenet --imagenet_path=<imagenet32x32 path>
```

For ImageNet 128x128 Conditional

```
python train.py --exp=imagenet_cond --num_steps=50 --batch_size=16 step_lr=100.0 --replay_batch --swish_act --cclass --zero_kl --dataset=imagenetfull --imagenet_datadir=<full imagenet path>
```

All code supports horovod execution, so model training can be speeded up substantially by using multiple different workers by running each command.
```
mpiexec -n <worker_num>  <command>
```

## Demo


The ebm_sandbox.py file contains several different tasks that can evaluate EBMs, by switching the task to different values.
For example, to visualize cross class mappings in CIFAR-10, you can run

```
python ebm_sandbox.py --task=crossclass --num_steps=40 --exp=cifar10_cond --resume_iter=74700
```


## Generalization

To test generalization to out of distribution classification run
```
python ebm_sandbox.py --task=gentest --num_steps=40 --exp=cifar10_cond --resume_iter=74700
```



## Concept Combination

To train each of conditional dsprites dataset, choose either cond_pos, cond_rot, cond_shape, cond_scale.

```
python train.py --dataset=dsprites --zero_kl --num_steps=20 --step_lr=500.0 --swish_act  --cond_pos --replay_batch
```

Once models are trained, they can be sampled from jointly by running

```
python ebm_combine.py --exp_size=<exp_size> --exp_shape=<exp_shape> --exp_pos=<exp_pos> --exp_rot=<exp_rot> --resume_size=<resume_size> --resume_shape=<resume_shape> --resume_rot=<resume_rot> --resume_pos=<resume_pos>
```



