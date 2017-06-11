import lasagne
import theano
from lasagne.layers import InputLayer
from lasagne.layers import DimshuffleLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from MeshGridLayer import MeshGridLayer
from AttentionInverseLayer import AttentionInverseLayer
from lasagne.layers import GRULayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ConcatLayer

from lasagne.nonlinearities import softmax, LeakyRectify


def build_model(image_size=100, detectors_per_window=9):
    glorot = lasagne.init.GlorotUniform()
    leaky = LeakyRectify(0.1)

    net = {}
    net['input'] = InputLayer((None, image_size, image_size))
    net['shuffled'] = DimshuffleLayer(net['input'], (0, 'x', 1, 2))

    # Convolutional layers
    net['conv1'] = Conv2DLayer(net['shuffled'], num_filters=32, filter_size=5, nonlinearity=leaky)
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=2)
    net['conv2'] = Conv2DLayer(net['pool1'], num_filters=48, filter_size=3, nonlinearity=leaky)
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=2)
    net['conv3'] = Conv2DLayer(net['pool2'], num_filters=64, filter_size=3, nonlinearity=leaky)
    net['pool3'] = MaxPool2DLayer(net['conv3'], pool_size=2)
    net['conv4'] = Conv2DLayer(net['pool3'], num_filters=86, filter_size=3, nonlinearity=leaky)
    net['pool4'] = MaxPool2DLayer(net['conv4'], pool_size=2)
    net['conv5'] = Conv2DLayer(net['pool4'], num_filters=128, filter_size=1, nonlinearity=leaky)
    net['conv6'] = Conv2DLayer(net['conv5'], num_filters=128, filter_size=2, stride=2, nonlinearity=leaky, pad='full')
    net['conv7'] = Conv2DLayer(net['conv6'], num_filters=128, filter_size=1, nonlinearity=leaky)

    # localise objects and make preliminary confidence scores:
    net['loc'] = Conv2DLayer(net['conv1x1b'], filter_size=1,
                            num_filters=4*detectors_per_window,
                            nonlinearity=lasagne.nonlinearities.identity) # B x 4D x H x W
    net['loc_global'] = AttentionInverseLayer(net['loc'], MeshGridLayer(net['loc'], detectors_per_window)) # transforming from local window coords to global image coords


    # filter out over-detected objects:
    net['concat'] = ConcatLayer([net['loc_global'],net['conv1x1b']])
    net['filter1'] = Conv2DLayer(net['concat'], num_filters=16*detectors_per_window, filter_size=3, pad='same')
    net['filter2'] = Conv2DLayer(net['filter1'], num_filters=16*detectors_per_window,filter_size=1)
    net['confidence'] = Conv2DLayer(net['filter2'], num_filters=detectors_per_window, filter_size=1, nonlinearity=lasagne.nonlinearities.sigmoid) # B x D x H x W

    # reshape everything into a nice output shape:
    net['loc_out'] = DimshuffleLayer(net['loc_global'], (0,2,3,1)) # B x H x W x 4D
    net['loc_out'] = ReshapeLayer(net['loc_out'], ([0], [1], [2], -1, 4)) # B x H x W x D x 4
    net['loc_out'] = ReshapeLayer(net['loc_out'], ([0],-1,4)) # B x HWD x 4

    net['confidence_out'] = DimshuffleLayer(net['confidence'], (0,2,3,1)) # B x H x W x D
    net['confidence_out'] = ReshapeLayer(net['confidence_out'], ([0], -1)) # B x HWD


    return net
