# avoiding-overdetection
Lasagne/Theano implementation of a fully convolutional neural network which attempts to output exactly one box per object, without requiring non-maximal suppression.

To create the dataset, go into `simcep`, launch Matlab and run `create_dataset.mat`. To adjust the parameters of the generated images (e.g. blur, illumination unevenness, clustering, overlap), edit `simcep_options_randomized.m`. To change the size of the images you'll need to edit both `create_dataset` and `simcep_options_randomized` (change `population.template`). To change the number of images, change `N` in `create_dataset`. The SIMCEP tool was created by [Antti Lehmussola](https://www.ncbi.nlm.nih.gov/pubmed/17649914), and was originally made available [here](http://www.cs.tut.fi/sgn/csb/simcep/tool.html).

Running `train.py` will create and train a model; when it finishes it will save the parameters, training logs and an evaluation figure in `results`. Most recent log file can be loaded with `openlogs.py`.
