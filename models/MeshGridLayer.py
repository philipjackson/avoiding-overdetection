import theano.tensor as T
import lasagne
import theano


class MeshGridLayer(lasagne.layers.Layer):
    """
    Takes a B x C x H x W tensor and outputs a B x HWD x 2 meshgrid. The
    meshgrid is 1 x 2 x H x W and is tiled up to B x 2D x H x W, then reshaped
    to B x HWD x 2. W is x-axis, H is y-axis. x and y values run from -1 to 1.
    """
    def __init__(self, incoming, D, **kwargs):
        super(MeshGridLayer, self).__init__(incoming, **kwargs)
        self.D = D
        if len(self.input_shape) != 4:
            raise ValueError("Incoming tensor must have 4 dimensions.")

    def get_output_shape_for(self, input_shapes):
        return (self.input_shape[0], 2*self.D, self.input_shape[2], self.input_shape[3])


    def get_output_for(self, input, **kwargs):
        batchsize = input.shape[0]
        input_shape = self.input_shape

        grid_x = T.mgrid[0:input_shape[2], 0:input_shape[3]][1].astype('float32') # H x W
        grid_x = (2 * grid_x / input_shape[3]) - 1
        grid_x = grid_x.reshape((1,1, grid_x.shape[0], grid_x.shape[1]), ndim=4) # 1 x 1 x H x W

        grid_y = T.mgrid[0:input_shape[2], 0:input_shape[3]][0].astype('float32')
        grid_y = (2 * grid_y / input_shape[2]) - 1
        grid_y = grid_y.reshape((1,1, grid_y.shape[0], grid_y.shape[1]), ndim=4)

        grid = T.concatenate([grid_x, grid_y], axis=1) # 1 x 2 x H x W
        zeros = T.zeros((1,2,grid.shape[2],grid.shape[3])) # 1 x 1 x H x W
        grid = T.concatenate([grid, zeros], axis=1) # 1 x 4 x H x W
        grid = T.tile(grid, (batchsize, self.D, 1, 1)) # B x 4D x H x W

        # grid = grid.dimshuffle((0,2,3,1))
        # grid = grid.reshape((grid.shape[0],grid.shape[1],grid.shape[2],self.D,2))
        # grid = grid.reshape((grid.shape[0],-1,2))

        return grid
