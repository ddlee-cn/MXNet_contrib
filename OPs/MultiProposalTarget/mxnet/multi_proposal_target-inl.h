/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * Copyright (c) 2017 Microsoft
 * Copyright (c) 2018 University of Maryland, College Park
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file multi_proposal_target-inl.h
 * \brief MultiProposalTarget Operator
*/

#ifndef MXNET_OPERATOR_MULTI_PROPOSAL_TARGET_INL_H_
#define MXNET_OPERATOR_MULTI_PROPOSAL_TARGET_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "operator_common.h"
#include "mshadow_op.h"


namespace mxnet {
namespace op {

namespace proposal {
enum MultiProposalTargetOpInputs {kClsProb, kBBoxPred, kImInfo, kGTBoxes, kValidRanges};
enum MultiProposalTargetOpOutputs {kRoIs, kLabels, kBboxTarget, kBboxWeight};
enum MultiProposalTargetForwardResource {kTempSpace};
}  // proposal

struct MultiProposalTargetParam : public dmlc::Parameter<MultiProposalTargetParam> {
  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float threshold;
  int rpn_min_size;
  int batch_size;
  float bbox_scale;
  uint64_t workspace;
  nnvm::Tuple<float> scales;
  nnvm::Tuple<float> ratios;
  int feature_stride;
  DMLC_DECLARE_PARAMETER(MultiProposalTargetParam) {
    float tmp[] = {0, 0, 0, 0, 0, 0, 0};
    DMLC_DECLARE_FIELD(rpn_pre_nms_top_n).set_default(6000)
    .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    DMLC_DECLARE_FIELD(rpn_post_nms_top_n).set_default(300)
    .describe("Overlap threshold used for non-maximum"
              "suppresion(suppress boxes with IoU >= this threshold");
    DMLC_DECLARE_FIELD(threshold).set_default(0.7)
    .describe("NMS value, below which to suppress.");
    DMLC_DECLARE_FIELD(batch_size).set_default(16)
    .describe("batch size");
    DMLC_DECLARE_FIELD(bbox_scale).set_default(1)
    .describe("Scale BBox Target");
    DMLC_DECLARE_FIELD(rpn_min_size).set_default(4)
    .describe("Minimum height or width in proposal");
    tmp[0] = 2.0f; tmp[1] = 4.0f; tmp[2] = 7.0f; tmp[3] = 10.0f; tmp[4] = 13.0f; tmp[5] = 16.0f; tmp[6] = 24.0f;
    DMLC_DECLARE_FIELD(scales).set_default(nnvm::Tuple<float>(tmp, tmp + 7))
    .describe("Used to generate anchor windows by enumerating scales");
    tmp[0] = 0.5f; tmp[1] = 1.0f; tmp[2] = 2.0f;
    DMLC_DECLARE_FIELD(ratios).set_default(nnvm::Tuple<float>(tmp, tmp + 3))
    .describe("Used to generate anchor windows by enumerating ratios");
    DMLC_DECLARE_FIELD(feature_stride).set_default(16)
    .describe("The size of the receptive field each unit in the convolution layer of the rpn,"
              "for example the product of all stride's prior to this layer.");
    DMLC_DECLARE_FIELD(workspace).set_default(128).set_range(0, 8192)
      .describe("Maximum temperal workspace allowed for kTempResource");
  }
};

template<typename xpu>
Operator *CreateOp(MultiProposalTargetParam param);

#if DMLC_USE_CXX11
class MultiProposalTargetProp : public OperatorProperty {
 public:
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
    CHECK_EQ(in_shape->size(), 5) << "Input:[cls_prob, bbox_pred, im_info, gt_boxes]";
    const TShape &dshape = in_shape->at(proposal::kClsProb);
    if (dshape.ndim() == 0) return false;

    aux_shape->clear();
    aux_shape->push_back(Shape2(dshape[0] * 21*32*32, 6));
    aux_shape->push_back(Shape2(dshape[0] * param_.rpn_post_nms_top_n, 5));    
    aux_shape->push_back(Shape2(21, 4));

    out_shape->clear();
    // output
    out_shape->push_back(Shape2(dshape[0] * param_.rpn_post_nms_top_n, 5));
    // label
    out_shape->push_back(Shape2(dshape[0] * param_.rpn_post_nms_top_n, 1));
    // bbox_target
    out_shape->push_back(Shape2(dshape[0] * param_.rpn_post_nms_top_n, 4));
    // bbox_weight
    out_shape->push_back(Shape2(dshape[0] * param_.rpn_post_nms_top_n, 4));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MultiProposalTargetProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MultiProposalTarget";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  int NumVisibleOutputs() const override {
    return 4;
  }

  int NumOutputs() const override {
    return 4;
  }

  std::vector<std::string> ListArguments() const override {
    return {"cls_prob", "bbox_pred", "im_info", "gt_boxes", "valid_ranges"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"rois", "label", "bbox_target", "bbox_weight"};
  }


  Operator* CreateOperator(Context ctx) const override;

 private:
  MultiProposalTargetParam param_;
};  // class MultiProposalProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_MULTI_PROPOSAL_INL_H_

