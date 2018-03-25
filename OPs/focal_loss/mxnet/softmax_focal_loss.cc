// Modified from Official Caffe2 implementation
// Author: ddlee, me@ddlee.cn

#include "./softmax_focal_loss-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftmaxFocalLossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
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

MXNET_REGISTER_OP_PROPERTY(SoftmaxFocalLoss, SoftmaxFocalLossProp)
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
.add_argument("data", "NDArray-or-Symbol", "4D tensor of softmax inputs (called 'scores' or 'logits') with shape (N, C, H, W), where C = num_anchors * num_classes defines num_anchors groups of contiguous num_classes softmax inputs.")
.add_argument("label", "NDArray-or-Symbol", "4D tensor of labels with shape (N, num_anchors, H, W). Each entry is a class label in [0, num_classes - 1] (inclusive).")
.add_argument("normalizer", "NDArray-or-Symbol"， "Scalar; the loss is normalized by 1 / max(1, normalizer).")
.add_arguments(SoftmaxFocalLossParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
