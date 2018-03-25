// Modified from Official Caffe2 implementation
// Author: ddlee, me@ddlee.cn

#include "./softmax_focal_loss-inl.h"


namespace mxnet {
    namespace op {
    
    template<>
    Operator* CreateOp<gpu>(SoftmaxFocalLossParam param, int dtype) {
      Operator* op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new SoftmaxFocalLossOp<gpu, DType>(param);
      });
      return op;
    }
    
    }  // namespace op
    }  // namespace mxnet