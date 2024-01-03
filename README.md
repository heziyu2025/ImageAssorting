# Image Assorting

## Installing
1. Clone repository
```cmd
git clone https://github.com/heziyu2025/ImageAssorting.git
cd ./ImageAssorting.git
```
2. Download and install [python](https://www.python.org/)
3. Install [pytorch](https://pytorch.org/) (If you have a nvidia gpu, you can use [cuda](https://developer.nvidia.com/cuda-toolkit) for higher speed)

## Training

To train the model, run:

```cmd
python3 ./train/train.py -e <epoch> -t <target> -f <file>
```

> `<epoch>` replace with the steps you want to train, `<target>` replace with the target correct rate, `<file>` replace with the file name.

## Usage

To run the model, run:

```cmd
python3 ./run_model.py -f <file>
```

> `<file>` replace your model file name.
