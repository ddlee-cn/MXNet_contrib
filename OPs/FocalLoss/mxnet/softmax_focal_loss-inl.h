/*
 * Author: ddlee, me@ddlee.cn
 * Modified from Official Caffe2 implementation
*/
// A multiclass form of Focal Loss designed for use in RetinaNet-like models.
// The input is assumed to be unnormalized scores (sometimes called 'logits')
// arranged in a 4D tensor with shape (N, C, H, W), where N is the number of
// elements in the batch, H and W are the height and width, and C = num_anchors *
// num_classes. The softmax is applied num_anchors times along the C axis.

// The softmax version of focal loss is:

//   FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t),

// where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
// s_j is the unnormalized score for class j.

// See: https://arxiv.org/abs/1708.02002 for details.

#ifndef MXNET_OPERATOR_SOFTMAX_FOCAL_LOSS_INL_H_
#define MXNET_OPERATOR_SOFTMAX_FOCAL_LOSS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include "../mshadow_op.h"
#include "../operator_common.h"

// namespace
namespace mxnet
{
namespace op
{

namespace focalloss
{
enum SoftmaxFocalLossOpInputs{kData, kLabel};
enum SoftmaxFocalLossOpOutputs{kLoss, kProb};
enum SoftmaxFocalLossOpResource{kTempSpace}; //
} // namespace focalloss

struct SoftmaxFocalLossParam : public dmlc::Parameter<SoftmaxFocalLossParam>
{
  float ignore_label;
  float grad_scale;
  float alpha;
  float gamma;
  DMLC_DECLARE_PARAMETER(SoftmaxFocalLossParam)
  {
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("The instances whose `labels` == `ignore_label` will be ignored "
              "during backward, if `use_ignore` is set to ``true``).");
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f).describe("(float) default 1.0; multiply the loss by this scale factor.");
    DMLC_DECLARE_FIELD(alpha).set_default(0.25f).describe("(float) default 0.25; Focal Loss's alpha hyper-parameter.");
    DMLC_DECLARE_FIELD(gamma).set_default(1.0f).describe("(float) default 1.0; Focal Loss's gamma hyper-parameter.");
  }
};

template <typename xpu, typename DType>
class SoftmaxFocalLossOp : public Operator
{
public:
  explicit SoftmaxFocalLossOp(SoftmaxFocalLossParam p)
  {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args)
  {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 2);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 3, DType> data = in_data[focalloss::kData].get<xpu, 3, DType>(s);
    Tensor<xpu, 2, DType> label = in_data[focalloss::kLabel].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> loss = out_data[focalloss::kLoss].get<xpu, 2, DType>(s);
    Tensor<xpu, 3, DType> prob = out_data[focalloss::kProb].get<xpu, 3, DType>(s);
  
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(label.CheckContiguous(), true);
    CHECK_EQ(prob.CheckContiguous(), true);

    SoftmaxFocalLossForward(data, label, loss, prob, param_.gamma, param_.alpha);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args)
  {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 2);

    Stream<xpu> *s = ctx.get_stream<xpu>();


    Tensor<xpu, 3, DType> data = in_data[focalloss::kData].get<xpu, 3, DType>(s);
    Tensor<xpu, 2, DType> label = in_data[focalloss::kLabel].get<xpu, 2, DType>(s);
    Tensor<xpu, 3, DType> prob = out_data[focalloss::kProb].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> grad_in = in_grad[focalloss::kData].get<xpu, 3, DType>(s); //dX
    Tensor<xpu, 2, DType> grad_out = out_grad[focalloss::kLoss].get<xpu, 2, DType>(s); //dloss

    Tensor<xpu, 2, DType> buff_ = ctx.requested[focalloss::kTempSpace]
      .get_space_typed<xpu, 2, DType>(label.shape_, s);
    
    buff_ = -1.0f;

    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(label.CheckContiguous(), true);
    CHECK_EQ(prob.CheckContiguous(), true);

    SoftmaxFocalLossBackwardAcc(data, label, prob, grad_in, grad_out, buff_, param_.gamma, param_.alpha);
  }

private:
  SoftmaxFocalLossParam param_;
}; // class SoftmaxFocalLossOp

// Decalre Factory function, used for dispatch specialization
template <typename xpu>
Operator *CreateOp(SoftmaxFocalLossParam param, int dtype);

#if DMLC_USE_CXX11
class SoftmaxFocalLossProp : public OperatorProperty
{
public:
  std::vector<std::string> ListArguments() const override
  {
    return {"data", "label"};
  }

  std::vector<std::string> ListOutputs() const override
  {
    return {"loss", "prob"};
  }

  int NumOutputs() const override
  {
    return 2;
  }

  int NumVisibleOutputs() const override
  {
    return 2;
  }

  void Init(const std::vector<std::pair<std::string, std::string>> &kwargs) override
  {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override
  {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override
  {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, label]";

    // data: (N, A, C) C =  num_class
    TShape dshape = in_shape->at(focalloss::kData);
    CHECK_EQ(dshape.ndim(), 3U) << "data should be a 3D tensor";

    // label: (N, A)
    TShape lshape = in_shape->at(focalloss::kLabel);
    CHECK_EQ(lshape.ndim(), 2U) << "label should be a 3D tensor";

    out_shape->clear();
    // loss: (N, A)
    out_shape->push_back(Shape2(dshape[0], dshape[1]));
    // prob: (N, A, C)
    out_shape->push_back(Shape3(dshape[0], dshape[1], dshape[2]));

    return true;
  }

  // bool InferType(std::vector<int> *in_type,
  //                std::vector<int> *out_type,
  //                std::vector<int> *aux_type) const override
  // {
  //   CHECK_EQ(in_type->size(), 2U);
  //   int dtype = (*in_type)[0];
  //   CHECK_EQ(dtype, (*in_type)[1]);
  //   CHECK_NE(dtype, -1) << "Input must have specified type";

  //   out_type->clear();
  //   out_type->push_back(dtype);
  //   out_type->push_back(dtype);

  //   aux_type->clear();
  //   aux_type->push_back(dtype);
  //   aux_type->push_back(dtype);
  //   return true;
  // }

  OperatorProperty *Copy() const override
  {
    SoftmaxFocalLossProp *softmax_focalloss_sym = new SoftmaxFocalLossProp();
    softmax_focalloss_sym->param_ = this->param_;
    return softmax_focalloss_sym;
  }

  std::string TypeString() const override
  {
    return "_contrib_SoftmaxFocalLoss";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override
  {
    return {out_grad[focalloss::kLoss], out_data[focalloss::kProb], in_data[focalloss::kData], in_data[focalloss::kLabel]};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator *CreateOperator(Context ctx) const override
  {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

private:
  SoftmaxFocalLossParam param_;
}; // class SoftmaxFocalLossProp
#endif
} // namespace op
} // namespace mxnet
#endif // MXNET_OPERATOR_SOFTMAX_FOCAL_LOSS_INL_H_
