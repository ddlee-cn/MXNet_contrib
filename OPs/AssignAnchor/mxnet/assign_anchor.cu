/*
 * Author: ddlee, me@ddlee.cn
 * modified from MXNet's MultiboxTarget Operator
*/

#include "./assign_anchor-inl.h"
#include <mshadow/cuda/tensor_gpu-inl.cuh>

#define ASSIGN_ANCHOR_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
namespace cuda {
template<typename DType>
__global__ void InitGroundTruthFlags(DType *gt_flags, const DType *labels,
                                     const int num_batches,
                                     const int num_labels,
                                     const int label_width) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_batches * num_labels) return;
  int b = index / num_labels;
  int l = index % num_labels;
  if (*(labels + b * num_labels * label_width + l * label_width) == -1.f) {
    *(gt_flags + b * num_labels + l) = 0; // dummy gt
  } else {
    *(gt_flags + b * num_labels + l) = 1; // need to be matched
  }
}

template<typename DType>
__global__ void FindBestMatches(DType *best_matches, DType *gt_flags, DType *gt_count,
                                DType *anchor_flags, DType *anchor_cls, const DType *labels,
                                const DType *overlaps, const int num_anchors,
                                const int num_labels, const int label_width) {
  int nbatch = blockIdx.x;
  gt_flags += nbatch * num_labels;
  gt_count += nbatch * num_labels;
  overlaps += nbatch * num_anchors * num_labels;
  best_matches += nbatch * num_anchors;
  anchor_flags += nbatch * num_anchors;
  labels += nbatch * num_labels * label_width;
  anchor_cls += nbatch * num_anchors;
  const int num_threads = kMaxThreadsPerBlock;
  __shared__ int max_indices_y[kMaxThreadsPerBlock];
  __shared__ int max_indices_x[kMaxThreadsPerBlock];
  __shared__ float max_values[kMaxThreadsPerBlock];

  while (1) {
    // check if all done.
    bool finished = true;
    for (int i = 0; i < num_labels; ++i) {
      if (gt_flags[i] > .5) {
        finished = false;
        break;
      }
    }
    if (finished) break;  // all done.

    // finding max indices in different threads
    int max_x = -1;
    int max_y = -1;
    DType max_value = 1e-6;  // start with very small overlap
    for (int i = threadIdx.x; i < num_anchors; i += num_threads) {
      if (anchor_flags[i] > .5) continue;
      for (int j = 0; j < num_labels; ++j) {
        if (gt_flags[j] > .5) {
          DType temp = overlaps[i * num_labels + j];
          if (temp > max_value) {
            max_x = j;
            max_y = i;
            max_value = temp;
          }
        }
      }
    }
    max_indices_x[threadIdx.x] = max_x;
    max_indices_y[threadIdx.x] = max_y;
    max_values[threadIdx.x] = max_value;
    __syncthreads();

    if (threadIdx.x == 0) {
      // merge results and assign best match, over all threads
      int max_x = -1;
      int max_y = -1;
      DType max_value = -1;
      for (int k = 0; k < num_threads; ++k) {
        if (max_indices_y[k] < 0 || max_indices_x[k] < 0) continue;
        float temp = max_values[k];
        if (temp > max_value) {
          max_x = max_indices_x[k];
          max_y = max_indices_y[k];
          max_value = temp;
        }
      }
      if (max_x >= 0 && max_y >= 0) {
        best_matches[max_y] = max_value;
        int offset_l = static_cast<int>(max_x) * label_width;
        anchor_cls[max_y] = labels[offset_l] + 1;
        // mark flags as visited
        // best match 
        // gt_count: -1 -> 0
        // anchor_flag: -1 -> 1
        gt_flags[max_x] = 0.f;
        gt_count[max_x] = 0.f;
        anchor_flags[max_y] = 1.f;
      } else {
        // no more good matches
        for (int i = 0; i < num_labels; ++i) {
          gt_flags[i] = 0.f;
        }
      }
    }
    __syncthreads();
  }
}

template<typename DType>
__global__ void FindGoodMatches(DType *best_matches, DType *anchor_flags,
                                DType *match, DType *anchor_cls, const DType *labels,
                                const DType *overlaps, const int num_anchors,
                                const int num_labels, const int label_width,
                                const float overlap_threshold) {
  int nbatch = blockIdx.x;
  overlaps += nbatch * num_anchors * num_labels;
  best_matches += nbatch * num_anchors;
  anchor_flags += nbatch * num_anchors;
  match += nbatch * num_anchors;
  anchor_cls += nbatch * num_anchors;
  labels += nbatch * num_labels * label_width;
  const int num_threads = kMaxThreadsPerBlock;

  for (int i = threadIdx.x; i < num_anchors; i += num_threads) {
    if (anchor_flags[i] < 0) {
      int idx = -1;
      float max_value = -1.f;
      for (int j = 0; j < num_labels; ++j) {
        DType temp = overlaps[i * num_labels + j];
        if (temp > max_value) {
          max_value = temp;
          idx = j;
        }
      }
      if (max_value > overlap_threshold && (idx >= 0)) {
        best_matches[i] = max_value;
        int offset_l = static_cast<int>(idx) * label_width;
        anchor_cls[i] = labels[offset_l] + 1;
        // good match
        // anchor_flag: -1 -> 0.9
        anchor_flags[i] = 0.9f;
        // cache good anchor matched label id
        match[i] = idx;
      }
    }
  }
}

template<typename DType>
__global__ void CollectGoodMatches(DType *gt_count,
                                    const DType *match, 
                                    const int num_anchors,
                                    const int num_labels){
  int nbatch = blockIdx.x;
  gt_count += nbatch * num_labels;
  match += nbatch * num_anchors;
  int idx = -1;
  for (int i = 0; i < num_anchors; i++){
    idx = int(match[i]);
    if (idx > -1){
      // accummulate each good match on that gt
      gt_count[idx] += 1.f;
    }
  }
}

}  // namespace cuda



template<typename DType>
inline void AssignAnchorForward(const Tensor<gpu, 2, DType> &anchor_flags_,
                           const Tensor<gpu, 2, DType> &best_matches_,
                           const Tensor<gpu, 2, DType> &gt_count_,
                           const Tensor<gpu, 2, DType> &anchor_cls_,
                           const Tensor<gpu, 2, DType> &anchors,
                           const Tensor<gpu, 3, DType> &labels,
                           const Tensor<gpu, 4, DType> &temp_space,
                           const float overlap_threshold) {
  const int num_batches = labels.size(0);
  const int num_labels = labels.size(1);
  const int label_width = labels.size(2);
  const int num_anchors = anchors.size(0);
  CHECK_GE(num_batches, 1);
  CHECK_GT(num_labels, 2);
  CHECK_GE(num_anchors, 1);

  temp_space[1] = -1.f;
  temp_space[2] = -1.f;
  DType *gt_flags = temp_space[1].dptr_;
  DType *match = temp_space[2].dptr_;
  DType *gt_count = gt_count_.dptr_;
  DType *anchor_flags = anchor_flags_.dptr_;
  DType *best_matches = best_matches_.dptr_;
  DType *anchor_cls = anchor_cls_.dptr_;

  // init ground-truth flags, by checking valid labels
  const int num_threads = cuda::kMaxThreadsPerBlock;
  dim3 init_thread_dim(num_threads);
  dim3 init_block_dim((num_batches * num_labels - 1) / num_threads + 1);
  cuda::CheckLaunchParam(init_block_dim, init_thread_dim, "AssignAnchor Init");
  cuda::InitGroundTruthFlags<DType><<<init_block_dim, init_thread_dim>>>(
    gt_flags, labels.dptr_, num_batches, num_labels, label_width);
  ASSIGN_ANCHOR_CUDA_CHECK(cudaPeekAtLastError());

  // compute best matches
  const DType *overlaps = temp_space[0].dptr_;
  cuda::CheckLaunchParam(num_batches, num_threads, "AssignAnchor Matching");
  cuda::FindBestMatches<DType><<<num_batches, num_threads>>>(best_matches,
    gt_flags, gt_count, anchor_flags, anchor_cls, labels.dptr_, overlaps, num_anchors, num_labels, label_width);
  ASSIGN_ANCHOR_CUDA_CHECK(cudaPeekAtLastError());

  // find good matches with overlap > threshold
  cuda::CheckLaunchParam(num_batches, num_threads, "AssignAnchor FindGood");
  cuda::FindGoodMatches<DType><<<num_batches, num_threads>>>(best_matches,
    anchor_flags, match, anchor_cls, labels.dptr_, overlaps, num_anchors, num_labels, label_width,
    overlap_threshold);
  ASSIGN_ANCHOR_CUDA_CHECK(cudaPeekAtLastError());


  cuda::CheckLaunchParam(num_batches, 1, "AssignAnchor Collect");
  cuda::CollectGoodMatches<DType><<<num_batches, 1>>>(gt_count, match, num_anchors, num_labels);
  ASSIGN_ANCHOR_CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(AssignAnchorParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AssignAnchorOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet
