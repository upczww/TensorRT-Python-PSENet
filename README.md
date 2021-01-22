This is a TensorRT implement of [Tensorflow_PSENet](https://github.com/liuheng92/tensorflow_PSENet)

## How to run
1. build `pse` algorithm
```
cd pse
make
```
2. prepare trained checkpoints in `./model`, likes
```
./model
 --checkpoint
 --model.ckpt-606001.data-00000-of-00001
 --model.ckpt-606001.index
 --model.ckpt-606001.meta
```

3. run main.py
```
python main.py --model_dir MODEL_DIR --engine_path ENGINE_PATH  --image_path IMAGE_PATH
```