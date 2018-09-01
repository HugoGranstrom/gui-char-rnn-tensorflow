GUI-char-rnn-tensorflow
===


GUI enabled Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Fork of sherjilozair's [char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow).
Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

## Requirements
- [Tensorflow 1.0](http://www.tensorflow.org)
- [Gooey](https://github.com/chriskiehl/Gooey)

# Everything beneath here is under review:
## Basic Usage
Start the program by running ``python main.py``

## Recommended starting parameters
``num_layers``: 2
``rnn_size``: 128
``model``: lstm
``batch_size``: 50 
``sequence_length``: 50
``num_epochs``: 50
``learning_rate``: 0.002


## Datasets
You can use any plain text file as input. For example you could download [The complete Sherlock Holmes](https://sherlock-holm.es/ascii/) as such:

```bash
cd data
mkdir sherlock
cd sherlock
wget https://sherlock-holm.es/stories/plain-text/cnus.txt
mv cnus.txt input.txt
```

A quick tip to concatenate many small disparate `.txt` files into one large training file: `ls *.txt | xargs -L 1 cat >> input.txt`

## Tensorboard
To visualize training progress, model graphs, and internal state histograms:  fire up Tensorboard and point it at your `log_dir`.  E.g.:
```bash
$ tensorboard --logdir=./logs/
```

Then open a browser to [http://localhost:6006](http://localhost:6006) or the correct IP/Port specified.


## Roadmap
- [ ] Add explanatory comments
- [ ] Expose more command-line arguments
- [ ] Compare accuracy and performance with char-rnn
- [ ] More Tensorboard instrumentation

## Contributing
Please feel free to:
* Leave feedback in the issues
* Open a Pull Request
* Join the [gittr chat](https://gitter.im/char-rnn-tensorflow/Lobby)
* Share your success stories and data sets!
