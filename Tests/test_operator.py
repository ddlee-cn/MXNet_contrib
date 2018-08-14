
import numpy as np
import mxnet as mx
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *



def np_softmax(x, axis=-1):
    # fix for old numpy on Travis not supporting keepdims
    # x = x - np.max(x, axis=-1, keepdims=True)
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x)
    # x /= np.sum(x, axis=-1, keepdims=True)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x

def cls2onehot(x, axis=-1):
    # convert class ids to one hot vector
    max_id = np.max(x)
    original_shape = list(x.shape)
    #print(original_shape)
    original_shape.insert(axis+1, max_id+1)
    #print(original_shape)
    zero_shape = list(x.flatten().shape)
    zero_shape.append(max_id+1)
    # generate zeros with shape (flatten, max_id+1)
    one_hot = np.zeros(zero_shape)
    one_hot[np.arange(zero_shape[0]), x.flatten()] = 1
    one_hot = np.reshape(one_hot, original_shape)
    return one_hot

def check_softmax_grad(xpu):
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.array([[1, 6, 4, 2]], ctx=xpu)
    grad_x = mx.nd.zeros((1,4), ctx=xpu)
    label_nd = mx.nd.array([1], ctx=xpu)

    sym = mx.sym.SoftmaxOutput(data=x, label=label, ignore_label=0, use_ignore=False)
    ex = sym.bind(ctx=xpu, args={'x': x_nd, 'label': label_nd}, args_grad={'x': grad_x})

    ex.forward(is_train=True)
    softmax_out = ex.outputs[0].asnumpy()
    expected_softmax_out = [[0.005806628, 0.861780069, 0.116629249, 0.015784052]]
    assert np.isclose(softmax_out, expected_softmax_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0].asnumpy()
    k = int(label_nd[0].asscalar())
    expected_grad_out = np.zeros((1,4))
    expected_grad_out[0, k] = -1
    assert np.isclose(grad_out - softmax_out, expected_grad_out).all()


def test_focal_loss_forward():
    """
    data : Symbol
        4D tensor of softmax inputs (called 'scores' or 'logits') with shape (N, C, H, W), where C = num_anchors * num_classes defines num_anchors groups of contiguous num_classes softmax inputs.
    label : Symbol
        4D tensor of labels with shape (N, num_anchors, H, W). Each entry is a class label in [0, num_classes - 1] (inclusive).
    normalizer : Symbol
        Scalar; the loss is normalized by 1 / max(1, normalizer).
    grad_scale : float, optional, default=1
        (float) default 1.0; multiply the loss by this scale factor.
    alpha : float, optional, default=0.25
        (float) default 0.25; Focal Loss's alpha hyper-parameter.
    gamma : float, optional, default=1
        (float) default 1.0; Focal Loss's gamma hyper-parameter.
    num_classes : int, optional, default='81'
        (int) default 81; number of classes in each softmax group.

    name : string, optional.
        Name of the resulting symbol.
    """
    np.random.seed(2018)
    # params
    epsilon = 1e-16
    alpha = 0.25
    gamma = 2
    num_anchors = 5 # number of anchors per location
    num_classes  = 12 # with background as index 0
    H = 3 #height
    W = 3 #width
    N = 4 #batch size
    C = num_anchors * num_classes
    wp = 20
    ctx = mx.gpu(0)

        # numpy array
    logits = np.random.normal(5, 1, [N, C, H, W])
    logits_reshape = logits.reshape([N, num_anchors, num_classes, H, W])
    label_data = np.random.randint(0, num_classes, [N, num_anchors, H, W])
    #label_one_hot = cls2onehot(label_data.reshape([N, num_anchors, H, W]), 1)

    prob = np_softmax(logits_reshape, axis=2) # calculate softmax output along classes axis
    # expected softmax output
    expected_softmax_prob = prob.reshape([N, C, H, W])

    preds = np.zeros([N, num_anchors, H, W])
    # take corresponding prob of class for CE loss
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            for u in range(preds.shape[2]):
                for v in range(preds.shape[3]):
                    cls = label_data[i][j][u][v]
                    preds[i][j][u][v] = prob[i][j][cls][u][v]
    alpha_ = np.where(label_data>=1, alpha, 1-alpha)

    expected_losses = -alpha_ * (1. - preds)**gamma * np.log(preds + epsilon)


    # mxnet ndarray
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    norm = mx.sym.Variable('norm')
    x_nd = mx.nd.array(logits, ctx=ctx)
    label_nd = mx.nd.array(label_data, ctx=ctx)
    norm_nd = mx.nd.array([wp], ctx=ctx)

    sym = mx.sym.contrib.SoftmaxFocalLoss(data=x, label=label, normalizer=norm, 
                                            gamma=gamma, alpha=alpha, num_classes=num_classes)
    arg_shapes, out_shapes, _ = sym.infer_shape(x=x_nd.shape, label=label_nd.shape, norm=norm_nd.shape)
    args_grad = [mx.nd.empty(s, ctx=ctx) for s in arg_shapes]
    ex = sym.bind(ctx=ctx, args={'x': x_nd, 'label': label_nd, 'norm': norm_nd}, args_grad=args_grad)
    ex.forward(is_train=True)
    focal_loss_out = ex.outputs[0].asnumpy()
    softmax_out = ex.outputs[1].asnumpy()

    assert_almost_equal(expected_softmax_prob, softmax_out)
    assert_almost_equal(expected_losses / wp, focal_loss_out)
    print('Softmax Focal Loss Forward Passed.')

def main():
    test_focal_loss_forward()

if __name__ == '__main__':
    main()