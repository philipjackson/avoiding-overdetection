import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

import numpy as np

def evaluate(batch, labels, infer_fn, test_fn, filename=None, pos_grad_fn=None, transform_params_fn=None, threshold=0.5, terminate=False):
    """ Evaluates the performance of infer_fn and test_fn on batch, with respect to ground-truth labels. Shows not just the cost but also plots the output locations for comparison against ground-truth."""
    # batch: images to evaluate on
    # labels: ground-truth labels for batch
    # infer_fn: theano function computing network forward pass
    # test_fn: theano function computing test loss (same network/params as infer_fn)
    # filename: if not None, save image to this filename
    assert(batch.shape[0] == labels.shape[0])
    n = batch.shape[0]
    rows = int(np.ceil(np.sqrt(n)))
    f, axarr = plt.subplots(rows,rows)
    for i in range(n):
        cost = test_fn(batch[i:i+1], labels[i:i+1])
        if isinstance(cost, list):
            cost = cost[0]

        try:
            ax = axarr[i//rows,i%rows]
        except:
            ax = axarr
        # ax.set_title("Cost: %f" % cost)
        ax.imshow(batch[i], cmap='gray')

        mask = labels[i,:,0] >= 0
        # ax.plot(labels[i,mask,0].flatten(), labels[i,mask,1].flatten(), 'og', fillstyle='none')
        detections, termination = infer_fn(batch[i:i+1])
        detections = detections.squeeze() # remove leading 1 dimension
        if detections.shape[1] == 4:
            boxes = True
            width, height = detections[:,2], detections[:,3]
            width = width*25
            height = height*25
            detections = detections[:,:2]
        else:
            boxes = False
        termination = termination.squeeze()
        if terminate:
            # using terminator neuron rather than per-output confidence
            termination = (1 - termination.cumsum()).clip(0,1)
        detections *= batch.shape[1:]
        # termination = np.cumsum(termination)
        # output_length = np.sum(termination < 0.5)
        print detections.shape
        if boxes:
            for j in range(len(detections)):

                if termination[j] < 0.1:
                    pass #continue
                verts = [(detections[j,0]-width[j]//2, detections[j,1]-height[j]//2),
                         (detections[j,0]+width[j]//2, detections[j,1]-height[j]//2),
                         (detections[j,0]+width[j]//2, detections[j,1]+height[j]//2),
                         (detections[j,0]-width[j]//2, detections[j,1]+height[j]//2)] # list of (x,y) tuples as expected by matplotlib
                verts.append(verts[0]) # closing the path
                codes = [Path.MOVETO,
                         Path.LINETO,
                         Path.LINETO,
                         Path.LINETO,
                         Path.CLOSEPOLY,
                         ] # copy/pasted from http://matplotlib.org/users/path_tutorial.html
                path = Path(verts, codes)
                edgecolor = 'red' if termination[j]>0.5 else 'blue'
                patch = patches.PathPatch(path, edgecolor=edgecolor, lw=1.0, alpha=termination[j], fill=False)
                ax.add_patch(patch)
            height, width = labels[i,:,2], labels[i,:,3]
            for j in range(len(labels[i])):
                break
                if labels[i,j,0] < 0:
                    continue
                verts = [(labels[i,j,0]-width[j]//2, labels[i,j,1]-height[j]//2),
                         (labels[i,j,0]+width[j]//2, labels[i,j,1]-height[j]//2),
                         (labels[i,j,0]+width[j]//2, labels[i,j,1]+height[j]//2),
                         (labels[i,j,0]-width[j]//2, labels[i,j,1]+height[j]//2)] # list of (x,y) tuples as expected by matplotlib
                verts.append(verts[0]) # closing the path
                codes = [Path.MOVETO,
                         Path.LINETO,
                         Path.LINETO,
                         Path.LINETO,
                         Path.CLOSEPOLY,
                         ] # copy/pasted from http://matplotlib.org/users/path_tutorial.html
                path = Path(verts, codes)
                edgecolor = "green"
                patch = patches.PathPatch(path, edgecolor=edgecolor, lw=0.6, fill=False)
                ax.add_patch(patch)
        else:
            ax.plot(detections[termination>=threshold,0].flatten(), detections[termination>=threshold,1].flatten(), 'rx')

        if pos_grad_fn:
            # plot gradients:
            grads = pos_grad_fn(batch[i:i+1], labels[i:i+1])
            grads = grads.squeeze()
            for j in range(output_length):
                ax.arrow(detections[j,0], detections[j,1], -grads[j,0], -grads[j,1])

        if transform_params_fn:
            # draw ROI boxes:
            for j in range(output_length):
                A = transform_params_fn(batch[i:i+1])[j]
                A = np.array([A[0],0,A[1],0,A[0],A[2]]).reshape((2,3))
                x = np.array([[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]).transpose()
                y = np.dot(A,x)
                y += 1
                y /= 2
                y = y * np.array([[batch.shape[2]],[batch.shape[1]]]) # re-scaling from normalised to image size
                print y

                verts = [tuple(point) for point in y.transpose()] # list of (x,y) tuples as expected by matplotlib
                verts.append(verts[0]) # closing the path
                codes = [Path.MOVETO,
                         Path.LINETO,
                         Path.LINETO,
                         Path.LINETO,
                         Path.CLOSEPOLY,
                         ] # copy/pasted from http://matplotlib.org/users/path_tutorial.html
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor='red', lw=1, alpha=0.5, fill=False)
                ax.add_patch(patch)


        ax.axis('off')

    if not filename:
        plt.show()
    else:
        plt.savefig(filename)

def average_count_error(images, targets, infer_fn, test_fraction=7.0):
    idx = range(int(len(images)-len(images)/test_fraction), len(images), 128)
    diff = []
    for i in idx:
        diff.extend(np.sum(targets[i:i+128,:,0]>=0,axis=1) - np.sum(infer_fn(images[i:i+128])[1]>=0.5,axis=1) )

    # return np.sqrt(np.mean(np.array(diff)**2))
    return np.mean(np.abs(diff))

def tpr(images, targets, infer_fn, test_fraction=7.0):
    positives = []
    tpr = []
    fpr = []
    precision = []
    for i in range(len(images)-int(len(images)/test_fraction)):
        positives.append(np.sum(targets[i,:,0]>0))
        target = targets[i]
        detections, confidence = infer_fn(images[i:i+1])
        detections = detections.squeeze()
        confidence = confidence.squeeze()
        detections = detections * np.array([224,224,25,25])
        iou = np.zeros((len(detections),positives[-1]))
        for j in range(iou.shape[0]):
            dx, dy, dw, dh = list(detections[j])
            for k in range(iou.shape[1]):
                tx, ty, tw, th = list(target[k])
                if tx < 0:
                    continue
                overlap = np.maximum(0,(np.minimum(dx+dw/2,tx+tw/2)-np.maximum(dx-dw/2,tx-tw/2)))*np.maximum(0,np.minimum(dy+dh/2,ty+th/2)-np.maximum(dy-dh/2,ty-th/2))
                iou[j,k] = overlap / (dw*dh+tw*th-overlap)

        cover = np.logical_and(iou>0.6, (confidence>0.5).reshape((-1,1)))
        tpr.append(0)
        fpr.append(0)
        precision.append(0)
        for k in range(iou.shape[1]):
            if np.any(cover[:,k]):
                j = np.where(cover[:,k])[0][0]
                cover[j,:] = False
                tpr[-1] += 1
        fpr[-1] = np.sum(confidence>0.5) - tpr[-1]
        if np.sum(confidence>0.5) > 0:
            precision[-1] = float(tpr[-1]) / np.sum(confidence>0.5)
        else:
            precision[-1] = 1
        # at this stage, tpr, fpr and tnr are counts rather than rates
        tpr[-1] = float(tpr[-1]) / positives[-1]
        fpr[-1] = float(fpr[-1]) / positives[-1]
        # if i == 17143+7:
        #     break
    return tpr, fpr, precision
