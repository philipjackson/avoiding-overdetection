import theano.tensor as T
import lasagne
import theano


class AttentionInverseLayer(lasagne.layers.MergeLayer):
    """
    Transforms a tensor of [0,1] two-vectors representing local positions within
    square attention windows back into global [0,1] image coordinates. Incoming
    must be B x m x 2 representing the local positions, meshgrid must be
    B x m x l where meshgrid[:,:,-2] is window x coords and meshgrid[:,:,-1] is
    window y coords.
    """
    def __init__(self, incoming, meshgrid, **kwargs):
        super(AttentionInverseLayer, self).__init__([incoming, meshgrid], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]


    def get_output_for(self, inputs, **kwargs):
        detections, meshgrid = inputs

        # # for correcting a BHWD x 2 location tensor:
        # # meshgrid is initially B x 2D x H x W
        # meshgrid = meshgrid.dimshuffle((0,2,3,1)) # B x H x W x 2D
        # meshgrid = meshgrid.reshape((meshgrid.shape[0], -1, meshgrid.shape[3])) # B x HW x 2D
        # meshgrid = meshgrid.reshape((meshgrid.shape[0], meshgrid.shape[1], meshgrid.shape[2]/2, 2)) # B x HW x D x 2
        # meshgrid = meshgrid.reshape((-1, 2)) # BHWD x 2
        # detections += (meshgrid+1) / 2

        # for correcting a B x 2D H x W location tensor:
        detections += (meshgrid+1)/2

        return detections
