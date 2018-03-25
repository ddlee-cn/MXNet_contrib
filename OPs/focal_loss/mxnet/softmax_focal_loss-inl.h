// Modified from Official Caffe2 implementation
// Author: ddlee, me@ddlee.cn


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
#include "./mshadow_op.h"
#include "./operator_common.h"

// namespace
namespace mxnet {
namespace op {

//TODO: ???
namespace softmax_focal_loss_enum {
enum SoftmaxFocalLossOpInputs {kData, kLabel, kNorm};
enum SoftmaxFocalLossOpOutputs {kLoss, kProb};
}  // namespace softmaxout_enum


struct SoftmaxFocalLossParam : public dmlc::Parameter< SoftmaxFocalLossParam> {
  float grad_scale;
  float alpha;
  float gamma;
  int num_classes;
  DMLC_DECLARE_PARAMETER(SoftmaxFocalLossParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("(float) default 1.0; multiply the loss by this scale factor.");
    DMLC_DECLARE_FIELD(alpha).set_default(0.25f)
    .describe("(float) default 0.25; Focal Loss's alpha hyper-parameter.");
    DMLC_DECLARE_FIELD(gamma).set_default(1.0f)
    .describe("(float) default 1.0; Focal Loss's gamma hyper-parameter.");
    DMLC_DECLARE_FIELD(num_classes).set_default(81)
    .describe("(int) default 81; number of classes in each softmax group.")
  };
};

template<typename xpu, typename DType>
class SoftmaxFocalLossOp : public Operator {
 public:
  explicit SoftmaxFocalLossOp(SoftmaxFocalLossParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);

    //TODO: shape check
    CHECK_EQ(out_data[focalloss::kOut].shape_[0], in_data[focalloss::kBox].shape_[0]);
    CHECK_EQ(out_data[focalloss::kMaxIdx].shape_[0], in_data[focalloss::kBox].shape_[0]);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[focalloss::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> labels = in_data[focalloss::kLabel].get<xpu, 4, DType>(s);
    Tensor<xpu, 1, DType> normalizer = in_data[focalloss::kNorm].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> loss = out_data[focalloss::kLoss].get<xpu, 1, DType>(s);
    Tensor<xpu, 4, DType> prob = out_data[focalloss::kProb].get<xpu, 4, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(labels.CheckContiguous(), true);
    CHECK_EQ(prob.CheckContiguous(), true);

    // TODO: Real Calculation
    focallossForward(out, data, bbox, max_idx, param_.spatial_scale);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 5);
    CHECK_EQ(out_data.size(), 1);
    
    // TODO: shape check
    CHECK_EQ(out_grad[focalloss::kOut].shape_[0], in_data[focalloss::kBox].shape_[0]);
    CHECK_EQ(out_data[focalloss::kMaxIdx].shape_[0], in_data[focalloss::kBox].shape_[0]);
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> grad_out = out_grad[focalloss::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad_in = in_grad[focalloss::kData].get<xpu, 4, DType>(s);

    Tensor<xpu, 4, DType> data = in_data[focalloss::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> labels = in_data[focalloss::kLabel].get<xpu, 4, DType>(s);
    Tensor<xpu, 1, DType> normalizer = in_data[focalloss::kNorm].get<xpu, 1, DType>(s);
    Tensor<xpu, 4, DType> prob = out_data[focalloss::kProb].get<xpu, 4, DType>(s);

    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(labels.CheckContiguous(), true);
    CHECK_EQ(prob.CheckContiguous(), true);

    // TODO: Real Calculation
    if (kAddTo == req[focalloss::kData] || kWriteTo == req[focalloss::kData]) {
      if (kWriteTo == req[focalloss::kData]) {
        grad_in = 0.0f;
      }
      focallossBackwardAcc(grad_in, grad_out, bbox, max_idx, param_.spatial_scale);
    }
    if (kWriteTo == req[focalloss::kBox]) {
      grad_roi = 0.0f;
    }
  }

 private:
  SoftmaxFocalLossParam param_;
};  // class SoftmaxFocalLossOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SoftmaxFocalLossParam param, int dtype);

#endif  // MXNET_OPERATOR_SOFTMAX_FOCAL_LOSS_INL_H_
