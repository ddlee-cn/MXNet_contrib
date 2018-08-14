/*
 * Author: ddlee, me@ddlee.cn
 * modified from MXNet's MultiboxTarget Operator
*/
#include <algorithm>
#include "./assign_anchor-inl.h"
#include "../mshadow_op.h"


namespace mshadow{
template<typename DType>
inline void AssignAnchorForward(const Tensor<cpu, 2, DType> &anchor_flags_,
                           const Tensor<cpu, 2, DType> &best_matches_,
                           const Tensor<cpu, 2, DType> &gt_count_,
                           const Tensor<cpu, 2, DType> &anchor_cls_,
                           const Tensor<cpu, 2, DType> &anchors,
                           const Tensor<cpu, 3, DType> &labels,
                           const Tensor<cpu, 4, DType> &temp_space,
                           const float overlap_threshold) {
    return;
}
}

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(AssignAnchorParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AssignAnchorOp<cpu, DType>(param);
  });
  return op;
}

Operator *AssignAnchorProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(AssignAnchorParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_AssignAnchor, AssignAnchorProp)
.describe(R"code(Assign Anchors among Ground Truths and collect assignments
Output:
Anchor_flags: nbatch * num_anchors, 0.9: good match, 1: best match, -1: no match
Best_matchs: nbatch * num_anchors, best IOU of every anchor
GT_count: nbatch * num_labels, 0: gt, with a best, >0: gt, with many good matches, -1: dummy(-1) gt
Anchor_cls: nbatch * num_anchors, matched gt class of every anchor
)code" ADD_FILELINE)
.add_argument("anchor", "NDArray-or-Symbol", "Generated anchor boxes.")
.add_argument("label", "NDArray-or-Symbol", "Object detection labels.")
.add_arguments(AssignAnchorParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
