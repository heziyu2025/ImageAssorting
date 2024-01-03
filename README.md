# Image Assorting

## Installing
1. Clone repository
```cmd
git clone https://github.com/heziyu2025/ImageAssorting.git
cd ./ImageAssorting
```
2. Download and install [python](https://www.python.org/)
3. Install [pytorch](https://pytorch.org/) (If you have a nvidia gpu, you can use [cuda](https://developer.nvidia.com/cuda-toolkit) for higher speed)

## Training

To train the model, run:

```cmd
py ./train/train.py -e <epoch> -t <target> -f <file>
```

> `<epoch>` replace with the steps you want to train, `<target>` replace with the target correct rate, `<file>` replace with the file name.
>
> e.g.:
>
> ```cmd
> py ./train/train.py -e 10 -t 0.8 -f my_model
> ```
>
> 

You can change the dataset in `train/dataset.py`.

## Usage

To run the model, run:

```cmd
py ./run_model.py -f <file> -i <index>
```

> `<file>` replace with your model file name, `<index>` replace with the index of the detaset image.
>
> e.g.:
>
> ```cmd
> py ./run_model.py -f my_model -i 0
> ```
