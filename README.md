# EBM

Code for reproducing results in <>

## Requirements

To install the prerequisites for the project run 
```
pip install -r requirements.txt
```

Download all saved models in the folder <>


## Image Generation

To generate images for CIFAR-10 and Imagenet run

All code supports horovod execution, so model training can be speeded up substantially by using multiple different workers.
```
mpiexec -n <worker_num>  <script>
```

## Image Manipulation

To test inpainting and other properties of the network down the pretrained model and run


## Generalization

To test generalization to out of distribution classification run
```
Code
```

## Concept Combination

To evaluate on the dSprites datasets, first download the dataset [here](https://github.com/deepmind/dsprites-dataset)

To train conditional models
```
python train.py
```
Models can be combined by

To test out of distribution generalization test.
