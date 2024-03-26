# evofc

### dependencies
* pytorch
* ptflops
* pymoo==0.3.0
* matplotlib
* lmdb
* tensorboardx
* nvidia-dali==1.32.0

or use conda:
```
conda env create -f environment.yml
```

### Prune resnet56 on cifar10
```
python search_prune.py --pop_size 10 --n_offsprings 10 --n_gens 20 --num_workers 4  --batch_size 128 --lr 1e-1 --arch resnet --depth 56 --epochs 300 --dataset cifar10

```

### Prune resnet34 on ILSVRC-2012
```
python search_prune.py --pop_size 40 --n_offsprings 40 --n_gens 30 --arch resnet --depth 34 --ws --epochs 100 --num_workers 4 --dataset imagenet --log-interval 100 --lr_scheduler step

```
### Prune mobilenetv2 on Places365-standard
```
python search_prune.py --pop_size 40 --n_offsprings 40 --n_gens 30 --arch mobilenetv2 --depth -1 --ws --epochs 300 --num_workers 4 --dataset places365 --log-interval 100 --lr_scheduler step --lr 0.045 --lr-gamma 0.98
```