/*
 * Author: ddlee, me@ddlee.cn
 * modified from MXNet's MultiboxTarget Operator
*/
#ifndef MXNET_OPERATOR_CONTRIB_ASSIGN_ANCHOR_INL_H_
#define MXNET_OPERATOR_CONTRIB_ASSIGN_ANCHOR_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include <nnvm/tuple.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <valarray>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace mshadow_op {
struct safe_divide {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    if (b == DType(0.0f)) return DType(0.0f);
    return DType(a / b);
  }
};  // struct safe_divide
}  // namespace mshadow_op

namespace AssignAnchor_enum {
enum AssignAnchorOpInputs {kAnchor, kLabel};
enum AssignAnchorOpOutputs {kAnchorFlag, kBestMatch, kGTCount, kAnchorCls};
enum AssignAnchorOpResource {kTempSpace};
}  // namespace AssignAnchor_enum

struct AssignAnchorParam : public dmlc::Parameter<AssignAnchorParam> {
  float overlap_threshold;
  DMLC_DECLARE_PARAMETER(AssignAnchorParam) {
    DMLC_DECLARE_FIELD(overlap_threshold).set_default(0.5f)
    .describe("Anchor-GT overlap threshold to be regarded as a positive match.");
  }
};  // struct AssignAnchorParam

template<typename xpu, typename DType>
class AssignAnchorOp : public Operator {
 public:
  explicit AssignAnchorOp(AssignAnchorParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow_op;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 4);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> anchors = in_data[AssignAnchor_enum::kAnchor]
      .get_with_shape<xpu, 2, DType>(
      Shape2(in_data[AssignAnchor_enum::kAnchor].size(1), 4), s);
    Tensor<xpu, 3, DType> labels = in_data[AssignAnchor_enum::kLabel]
      .get<xpu, 3, DType>(s);
    Tensor<xpu, 2, DType> anchor_flag = out_data[AssignAnchor_enum::kAnchorFlag]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> best_match = out_data[AssignAnchor_enum::kBestMatch]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> gt_count = out_data[AssignAnchor_enum::kGTCount]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> anchor_cls = out_data[AssignAnchor_enum::kAnchorCls]
      .get<xpu, 2, DType>(s);

    index_t num_batches = labels.size(0);
    index_t num_anchors = anchors.size(0);
    index_t num_labels = labels.size(1);
    // TODO(zhreshold): use maximum valid ground-truth in batch rather than # in dataset
    Shape<4> temp_shape = Shape4(11, num_batches, num_anchors, num_labels);
    Tensor<xpu, 4, DType> temp_space = ctx.requested[AssignAnchor_enum::kTempSpace]
      .get_space_typed<xpu, 4, DType>(temp_shape, s);


    anchor_flag = -1.f;
    best_match = -1.0f;
    gt_count = -1.f;
    temp_space = -1.0f;
    anchor_cls = -1.f;
    CHECK_EQ(anchors.CheckContiguous(), true);
    CHECK_EQ(labels.CheckContiguous(), true);
    CHECK_EQ(anchor_flag.CheckContiguous(), true);
    CHECK_EQ(best_match.CheckContiguous(), true);
    CHECK_EQ(gt_count.CheckContiguous(), true);
    CHECK_EQ(temp_space.CheckContiguous(), true);
    CHECK_EQ(anchor_cls.CheckContiguous(), true);

    // compute overlaps
    // TODO(zhreshold): squeeze temporary memory space
    // temp_space, 0:out, 1:l1, 2:t1, 3:r1, 4:b1, 5:l2, 6:t2, 7:r2, 8:b2
    // 9: intersection, 10:union
    temp_space[1] = broadcast_keepdim(broadcast_with_axis(slice<1>(anchors, 0, 1), -1,
      num_batches), 2, num_labels);
    temp_space[2] = broadcast_keepdim(broadcast_with_axis(slice<1>(anchors, 1, 2), -1,
      num_batches), 2, num_labels);
    temp_space[3] = broadcast_keepdim(broadcast_with_axis(slice<1>(anchors, 2, 3), -1,
      num_batches), 2, num_labels);
    temp_space[4] = broadcast_keepdim(broadcast_with_axis(slice<1>(anchors, 3, 4), -1,
      num_batches), 2, num_labels);
    Shape<3> temp_reshape = Shape3(num_batches, 1, num_labels);
    temp_space[5] = broadcast_keepdim(reshape(slice<2>(labels, 1, 2), temp_reshape), 1,
      num_anchors);
    temp_space[6] = broadcast_keepdim(reshape(slice<2>(labels, 2, 3), temp_reshape), 1,
      num_anchors);
    temp_space[7] = broadcast_keepdim(reshape(slice<2>(labels, 3, 4), temp_reshape), 1,
      num_anchors);
    temp_space[8] = broadcast_keepdim(reshape(slice<2>(labels, 4, 5), temp_reshape), 1,
      num_anchors);
    temp_space[9] = F<maximum>(ScalarExp<DType>(0.0f),
      F<minimum>(temp_space[3], temp_space[7]) - F<maximum>(temp_space[1], temp_space[5]))
        * F<maximum>(ScalarExp<DType>(0.0f),
        F<minimum>(temp_space[4], temp_space[8]) - F<maximum>(temp_space[2], temp_space[6]));
    temp_space[10] = (temp_space[3] - temp_space[1]) * (temp_space[4] - temp_space[2])
     + (temp_space[7] - temp_space[5]) * (temp_space[8] - temp_space[6])
      - temp_space[9];
    temp_space[0] = F<safe_divide>(temp_space[9], temp_space[10]);

    AssignAnchorForward(anchor_flag, best_match, gt_count, anchor_cls,
                          anchors, labels, temp_space,
                          param_.overlap_threshold);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
                          // no implementation
}

 private:
  AssignAnchorParam param_;
};  // class AssignAnchorOp

template<typename xpu>
Operator* CreateOp(AssignAnchorParam param, int dtype);

#if DMLC_USE_CXX11
class AssignAnchorProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"anchor", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"anchor_flag", "best_match", "gt_count", "anchor_cls"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input: [anchor, label]";
    TShape ashape = in_shape->at(AssignAnchor_enum::kAnchor);
    CHECK_EQ(ashape.ndim(), 3) << "Anchor should be batch shared N*4 tensor";
    CHECK_EQ(ashape[0], 1) << "Anchors are shared across batches, first dim=1";
    CHECK_GT(ashape[1], 0) << "Number boxes should > 0";
    CHECK_EQ(ashape[2], 4) << "Box dimension should be 4: [xmin-ymin-xmax-ymax]";
    TShape lshape = in_shape->at(AssignAnchor_enum::kLabel);
    CHECK_EQ(lshape.ndim(), 3) << "Label should be [batch-num_labels-(>=5)] tensor";
    CHECK_GT(lshape[1], 0) << "Padded label should > 0";
    CHECK_GE(lshape[2], 5) << "Label width must >=5";
    // all output should have shape (batch, num_box)
    TShape anchor_flag_shape = Shape2(lshape[0], ashape[1]);
    TShape gt_flag_shape = Shape2(lshape[0], lshape[1]);
    TShape best_match_shape = Shape2(lshape[0], ashape[1]);
    TShape anchor_cls_shape = Shape2(lshape[0], ashape[1]);
    out_shape->clear();
    out_shape->push_back(anchor_flag_shape);
    out_shape->push_back(best_match_shape);
    out_shape->push_back(gt_flag_shape);
    out_shape->push_back(anchor_cls_shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    AssignAnchorProp* AssignAnchor_sym = new AssignAnchorProp();
    AssignAnchor_sym->param_ = this->param_;
    return AssignAnchor_sym;
  }

  std::string TypeString() const override {
    return "_contrib_AssignAnchor";
  }

  //  decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  std::vector<ResourceRequest> ForwardResource(
       const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                              std::vector<int> *in_type) const override;

 private:
  AssignAnchorParam param_;
};  // class AssignAnchorProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_ASSIGN_ANCHOR_INL_H_
