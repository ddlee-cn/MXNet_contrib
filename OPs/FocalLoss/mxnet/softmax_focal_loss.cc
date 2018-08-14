/*
 * Author: ddlee, me@ddlee.cn
 * Modified from Official Caffe2 implementation
*/
#include "./softmax_focal_loss-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include "../mshadow_op.h"

namespace mshadow {
  template<typename DType>
    inline void SoftmaxFocalLossForward(const Tensor<cpu, 3, DType> &X, // Logits; data
                                        const Tensor<cpu, 2, DType> &T, // Labels; labels
                                        const Tensor<cpu, 2, DType> &loss, // aux losses_ Tensor
                                        const Tensor<cpu, 3, DType> &P, //softmax probability, going to be re-used in gradient; prob
                                        const float gamma_,
                                        const float alpha_)
  {
      // not implemented
      return;
  };

  template<typename DType>
    inline void SoftmaxFocalLossBackwardAcc(const Tensor<cpu, 3, DType> &X, // Logits; data
                                            const Tensor<cpu, 2, DType> &T, // Labels; labels
                                            const Tensor<cpu, 3, DType> &P, //softmax probability; prob
                                            const Tensor<cpu, 3, DType> &dX, // gradient out
                                            const Tensor<cpu, 2, DType> &dloss, // gradient in
                                            const Tensor<cpu, 2, DType> &buff_, // aux buff_ Tensor
                                            const float gamma_,
                                            const float alpha_)
  {
      // not implemented
      return;
  };
}


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftmaxFocalLossParam param, int DType) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(DType, DType, {
    op = new SoftmaxFocalLossOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SoftmaxFocalLossProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SoftmaxFocalLossParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_SoftmaxFocalLoss, SoftmaxFocalLossProp)
.describe(R"code(A multiclass form of Focal Loss designed for use in RetinaNet-like models.
The input is assumed to be unnormalized scores (sometimes called 'logits')
arranged in a 4D tensor with shape (N, C, H, W), where N is the number of
elements in the batch, H and W are the height and width, and C = num_anchors *
num_classes. The softmax is applied num_anchors times along the C axis.

The softmax version of focal loss is:

  FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t),

where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
s_j is the unnormalized score for class j.

See: https://arxiv.org/abs/1708.02002 for details.
)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "3D tensor of softmax inputs (called 'scores' or 'logits') with shape (N, A, C), where C = num_anchors * num_classes defines num_anchors groups of contiguous num_classes softmax inputs.")
.add_argument("label", "NDArray-or-Symbol", "2D tensor of labels with shape (N, A). Each entry is a class label in [0, num_classes - 1] (inclusive).")
.add_arguments(SoftmaxFocalLossParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
