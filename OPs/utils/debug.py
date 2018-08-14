import mxnet as mx
import time
import numpy as np


class Debug(mx.operator.CustomOp):
    def forward(self, req, in_data, out_data, aux, is_train=False):
        value = in_data[0].asnumpy()
        # print('in_data_shape', value.shape)
        # print('in_data', value)
        # time.sleep(1)
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass
        # self.assign(in_grad[0],req[0], out_grad[0])


@mx.operator.register("debug")
class DebugProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(DebugProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        args_list = ['data']
        return args_list

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        print(in_shape)
        output_shape = in_shape
        return in_shape, output_shape

    def create_operator(self, ctx, shapes, dtypes):
        return Debug()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
