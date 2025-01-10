
# Fine tuning network weights

This tutorial describes the procedure of fine tuning (i.e., optimizing existing trained weights) a neural network for instance segmentation.
The instructions to train a model from scratch are very similar, and can be inferred from this tutorial, the only exception is that you would start with untrained network.

We consider the scenario where high-quality labeled data isn't available.
Hence, we train our network from labels computed through a classical image processing pipeline (e.g. watershed).
You might wonder why we need to train a DL model to predict results if the watershed results are used as our training data, such that, we expect to produce results as good as the watershed results.
That's a valid point, it's beneficial because we're more concerned with the topology (i.e. basins like) of the image, which is somewhat messy in the original image and can be improved with the neural network.
Moreover, the model inductive bias produces smoother outputs, filtering some of the noise.

## Data preparation

Starting from a DEXP's dataset we need to obtain our "weak" labels. To do that we provide a sample script named `segment.py` that creates a directory named `weak_labels` with the image frames and their respective labels.

This script filters the image to detect the cells and segment them using variant of the h-watershed _[1] that penalizes pixels further away from the minimum, resulting in more convex-like segments. In summary, it's a hybrid of watershed and voroni.

We recommend using the `--display` flag first to pick best segmentation parameters and executing without it to process the whole dataset.

The step size `-s` controls the step to select frames over the t-axis, and it should be chosen according to how much training data you want.
You should take into account that later we're going split these frames into smaller tiles for training, and a single stack of shape (512, 2048, 2048) can generate more than 2400 tiles of shape (96, 96, 96). However, some tiles are empty and discarted, and the are sampled randomly, so they could overlap each other.

```bash
python segment.py -i <path to dexp dataset .zarr> -o weak_labels -c fused -s 10 -z 2 --display
```

The sample code `segment.py` is:

.. literalinclude:: code/segment.py

The next step is to split these frames into smaller tiles (i.e. chunks) for training. DEXP-DL provides a helper command for that, `tif2tiles`.
We achieved good results with 1000 tiles, but this is not a rule, the shape of the tile should vary according to your available GPU memory, the bigger the better.

Don't forget the quotation marks around the paths with `*`.

```bash
dexp-dl tif2tiles -i 'weak_labels/images/*.tif' -l 'weak_labels/labels/*.tif' -o tiles -n 1000 -t [1,96,96,96]
```


## Fine tuning

Finally, we fine-tune our DL model using the scripts located at `dexp-dl/examples/training`.
For a dictory full of matching tiles the command should be called as below, feel free to decrease or increase the number of epochs (iterations).

```bash
python <path to dexp-dl>/examples/training/edge_detection_3d_tiles.py -d tiles -n fine-tuning -e 10 -w <path to existing weights.ckpt>
```

To monitor the training we recommend using `tensorboard`. For that, on another terminal window execute:

```bash
tensorboard --logdir logs
```

The `logs` directory is created automatically on the working directory of the training script.

The new weights can be found at `logs/last.ckpt` when the training is over.
Usually we rename the weights to keep track of when it was created and from what dataset.

From this `.ckpt` file you can compute new predictions using our `dexp-dl/examples/inference` scripts.


## References

[1] Falcão, Alexandre X., Jorge Stolfi, and Roberto de Alencar Lotufo. "The image foresting transform:
    Theory, algorithms, and applications." IEEE transactions on pattern analysis and
    machine intelligence 26.1 (2004): 19-29.
