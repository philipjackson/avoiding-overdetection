import os

import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import l2, regularize_network_params, regularize_layer_params

import numpy as np
import pickle, os, sys, glob, time, argparse
import matplotlib.pyplot as plt
import evaluate

num_epochs = 200
batchsize = 128
test_fraction = 7 # 1/test_fraction images will be used for testing
learning_rate = 0.0001
weight_decay = 5 * 1e-5
MAX_NORM = 1.0

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", default="datasets/simcep_224px_15cells.mat")
args = parser.parse_args()

def build_model(input_var, model='filter-cnn', pretrain=True):
    if 'models.filter_cnn' in globals():
        reload(models.filter_cnn)
    else:
        import models.filter_cnn
    network, sh_drp = models.filter_cnn.build_model(image_size=images.shape[-1])
    network['input'].input_var = input_var

    return network

def load_dataset(filename='datasets/simcep_224px_15cells.mat'):
    # returns hdf5 references to the dataset
    print "Loading dataset... ",
    sys.stdout.flush()

    import h5py
    matfile = h5py.File(filename,'r')

    images = matfile['images'][:]
    # convert to the correct type:
    images = images.astype(theano.config.floatX)
    # normalise to 0 mean unit variance:
    images = (images - images.mean()) / images.std()

    # cell coordinates require some munging
    # first, extract them from the file as arrays (technically h5py datasets):
    targets = []
    for i in range(len(images)):
        array = matfile[matfile['gt'][i,0]][:] # de-references a reference to a dataset, then reads that dataset into an np.ndarray
        targets.append(array.flatten())        # here we flatten the array from a single column matrix to a vector
    # at this point targets is a list of 4*n[i] length vectors, where n[i] is the number of cells in image i
    # now pad all sequences to max length:
    maxlen = max(map(len, targets))
    targets = map(lambda s:np.lib.pad(s, (0,maxlen-len(s)), 'constant', constant_values=-2*np.hypot(*images.shape[1:])), targets) # pad with minus twice the diagonal length of the images. This ensures that no detection will be closer to a padded pseudo-point than to a real point (unless the image contains no cells...)
    assert(maxlen % 4 == 0) # each item in targets should be a flattened (maxlen x 4) matrix
    assert(map(len, targets).count(maxlen) == len(targets)) # make sure they're all the same shape (should be due to padding)
    # un-flatten the coordinate axes:
    targets = map(lambda s:s.reshape((maxlen/4,4)), targets)
    assert(np.all(map(lambda s:s.shape==(maxlen/4,4), targets)))
    # concatenate sequences into a single rank 3 tensor:
    targets = np.stack(targets)
    targets = targets.astype(theano.config.floatX)

    print "Done"
    sys.stdout.flush()

    return images, targets

def iterate_minibatches(images, targets, batchsize, shuffle=True, training=True):
    """
    images:    ndarray-like, shape N x H x W
    targets:   ndarray-like, shape N x MAXLEN x 2
    batchsize: images per minibatch
    shuffle:   will iterate through dataset in random order if true
    training:  returns images/targets from training set if true, test set if false
               (images and targets partitioned according to the global float test_fraction)
    """
    import random

    N = len(images)
    N_test = N // test_fraction
    N_train = N - N_test

    if training:
        indices = range(N_train)
    else:
        indices = range(N_train, N)
    if shuffle:
        random.shuffle(indices)

    for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        excerpt.sort()          # necessary to keep h5py happy
        batch = images[excerpt].astype('float32') # just in case
        labels = targets[excerpt]

        yield batch, labels


# ================================ MAIN PROGRAM ================================
if not ('images' in globals() and 'targets' in globals()):
    images, targets = load_dataset(args.dataset)
    maxlen = targets.shape[1]

print "Building model, compiling functions... ",
sys.stdout.flush()

input_var = T.tensor3('input_var')   # batch of images (float32 batchsize x 224 x 224)
target_var = T.tensor3('target_var') # batch of labels (float32 batchsize x maxlen x 2)
# mask_var = target_var[:,:,0] >= 0    # batch of input masks (float32 batchsize x maxlen)

network = build_model(input_var, model=args.model, pretrain=args.load_params)

# SETUP COST
def cost_mindist_fun(detections, end):
    """
    Closest detections to GT points are pulled closer and made more confident.
    Other detections have their position unaltered, but are made less confident.
    """

    # normalize outputs from [0,1] -> image shape:
    detections = detections * T.concatenate([input_var.shape[1:],[25,25]],axis=0)
    # some things we'll need later:
    n_steps = detections.shape[1]
    mask = target_var[:,:,0] >= 0
    diagonal_length = T.sqrt(input_var.shape[-1]**2 + input_var.shape[-2]**2)

    # get square distance matrices:
    distmat = T.sqr(detections).sum(axis=2).reshape((-1, n_steps, 1)) + \
              T.sqr(target_var).sum(axis=2).reshape((-1, 1, maxlen))
    distmat = distmat - 2 * T.batched_dot(detections, target_var.dimshuffle((0,2,1)))
    # distmat has shape (batchsize x maxlen x maxlen)

    # regression loss:
    mindist_loss = distmat.min(axis=1) / diagonal_length # batchsize x maxlen
    # mask it out to avoid padding cells contributing to regression loss:
    mindist_loss = (mindist_loss*mask).sum(axis=1) / mask.sum(axis=1)

    # confidence loss:
    responsibility = confidence.dimshuffle((0,1,'x')) / (distmat+1)
    nearest_idx = responsibility.argmax(axis=1) # batchsize x maxlen
    dummy_idx = nearest_idx[:,0].dimshuffle((0,'x')) # stop padding cells from setting any detection's target confidence to 1
    nearest_idx = mask * nearest_idx + (1-mask) * dummy_idx
    confidence_target = T.zeros(end.shape)
    confidence_subtensor = confidence_target[T.arange(end.shape[0]).dimshuffle((0,'x')), nearest_idx]
    confidence_target = T.set_subtensor(confidence_subtensor, 1.) # cheers nowiz
    confidence_loss = (end - confidence_target)**2 # batchsize x n_steps
    confidence_loss = confidence_loss.mean(axis=1)

    # assemble:
    cost = mindist_loss + confidence_loss
    return cost.mean(), mindist_loss.mean(), confidence_loss.mean(), responsibility

# get network output sequence (batchsize x n_steps x 2):
detections = lasagne.layers.get_output(network['loc_out'], deterministic=False)
# and the confidence vector (batchsize x n_steps):
confidence = lasagne.layers.get_output(network['confidence_out'], deterministic=False)
# and again for test time (same but with dropout disabled):
detections_test = lasagne.layers.get_output(network['loc_out'], deterministic=True)
confidence_test = lasagne.layers.get_output(network['confidence_out'], deterministic=True)

# calculate cost:
cost, cost_mindist, cost_confidence, responsibility = cost_mindist_fun(detections, confidence)
cost_test, cost_mindist_test, cost_confidence_test, responsibility_test = cost_mindist_fun(detections_test, confidence_test)

# add L2 regularization (aka weight decay):
l2_penalty = regularize_network_params([network['loc_out'], network['confidence_out']], l2) * weight_decay
cost = cost + l2_penalty

# define updates:
params = lasagne.layers.get_all_params([network['loc_out'], network['confidence_out']],
                                       trainable=True)
all_grads = T.grad(cost, params)
all_grads = [T.clip(g, -1, 1) for g in all_grads]
updates, norm = lasagne.updates.total_norm_constraint(
    all_grads, max_norm=MAX_NORM, return_norm=True)
updates = lasagne.updates.rmsprop(updates, params, learning_rate=learning_rate, rho=0.9)
pos_grad = T.grad(cost, detections)

# Compile functions:
train_fn = theano.function([input_var, target_var], [cost-l2_penalty, cost_mindist, cost_confidence, l2_penalty], updates=updates)
test_fn = theano.function([input_var, target_var], [cost_test, cost_mindist_test, cost_confidence_test])

infer_fn = theano.function([input_var], [detections_test, confidence_test])
pos_grad_fn = theano.function([input_var, target_var], pos_grad)

print "Done"
sys.stdout.flush()

N = len(images)
N_test = N//test_fraction
N_train = N - N_test


log = {'training_cost':[],'training_cost_mindist':[],'training_cost_confidence':[],'training_cost_l2':[],'test_cost':[],'test_cost_mindist':[],'test_cost_confidence':[]}
# l2_loss = []
t00 = time.time()
for i in range(num_epochs):
    # train on the full training set:
    error_training = 0
    batches_training = 0
    t0 = time.time()
    print "\n============================ Epoch %d ============================" % (i,)
    print "Cost \t \t Cost_mindist \t Cost_confidence \t Cost_L2"


    for batch, labels in iterate_minibatches(images, targets, batchsize, shuffle=True, training=True):
        error_current, cost_mindist_current, cost_confidence_current, cost_l2_current = train_fn(batch, labels)
        print "%f \t %f \t %f \t %f\n" % (error_current, cost_mindist_current, cost_confidence_current, cost_l2_current),
        sys.stdout.flush()
        error_training += error_current

        log['training_cost'].append(error_current)
        log['training_cost_l2'].append(cost_l2_current)
        log['training_cost_mindist'].append(cost_mindist_current)
        log['training_cost_confidence'].append(cost_confidence_current)

        batches_training += 1

    print "Epoch %d took %f seconds to train on %d images" % (i,time.time()-t0, N_train)
    sys.stdout.flush()

    # test on the full testing set:
    error_testing = 0
    batches_testing = 0
    for batch, labels in iterate_minibatches(images, targets, batchsize, shuffle=True, training=False):
        error_testing_current, error_testing_mindist_current, error_testing_confidence_current = test_fn(batch, labels)
        error_testing += error_testing_current

        log['test_cost'].append(error_testing_current)
        log['test_cost_mindist'].append(error_testing_mindist_current)
        log['test_cost_confidence'].append(error_testing_confidence_current)
        batches_testing += 1
    print "Epoch %d: mean training loss = %f, mean testing loss = %f\n" % \
             (i, error_training/batches_training, error_testing/batches_testing)
    sys.stdout.flush()

print "In total, training and testing took %f seconds" % (time.time()-t00,)
print "Average absolute test count error:", evaluate.average_count_error(images,targets,infer_fn)
sys.stdout.flush()

timestr = time.strftime("%d-%m-%Y_%H:%M")

try:
    filename = "results/" + args.model + "_params_" + timestr+".pkl"
    fout = open(filename, 'wb')
    params = lasagne.layers.get_all_param_values([network['loc_out'], network['confidence_out']])
    pickle.dump(params, fout)
    fout.close()
    print "Saved params as", filename
except Exception, e:
    print "Failed to pickle params"
    print str(e)

try:
    filename = "results/" + args.model + "_log_" + timestr +".pkl"
    fout = open(filename, 'w')
    pickle.dump(log, fout)
    fout.close()
    print "Saved training logs as", filename
except Exception, e:
    print "Failed to save training logs"
    print str(e)

try:
    filename = "results/" + args.model + "_fig_" + timestr + ".png"
    evaluate.evaluate(images[0:16], targets[0:16], infer_fn, test_fn, filename=filename)
    print "Saved evaluation figure as", filename
except Exception, e:
    print "Failed to save figure"
    print str(e)
