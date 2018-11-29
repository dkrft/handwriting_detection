installation:

```bash
sudo pacman -S python-tensorflow-cuda
```

https://www.tensorflow.org/guide/using_gpu

to get a list of devices that you can run tensorflow on:

```bash
python -c "import tensorflow as tf; tf.Session(config=tf.ConfigProto(log_device_placement=True))" 2>&1 >/dev/null | grep -P "^/" | grep -P "/device:.*? "
```

example: `/device:XLA_GPU:0`

I wrote this value into hwdetect/neural_network/model.py into the following line in the function 'create':

```python
with tf.device('/device:XLA_GPU:0'):
```

but then I fugred out my cpu is still full load (don't remember if it was faster)

after rebooting It seemed to work though.

However, then I figured out, it always uses the gpu now despite the tf.device.

The real thing to set is in hwdetect/neural_network/predictor.py:

https://stackoverflow.com/questions/37660312/how-to-run-tensorflow-on-cpu

GPU:
```python
config = tf.ConfigProto(device_count = {'GPU': 1})
```

CPU:
```python
config = tf.ConfigProto(device_count = {'GPU': 0})
```

So I added a parameter to Predictor called gpu, that is either 0 or 1
