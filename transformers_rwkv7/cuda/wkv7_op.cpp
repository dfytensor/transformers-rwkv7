// C++ wrapper for the enhanced WKV-7 kernel. Registers a torch op that dispatches by dtype.
#include <torch/extension.h>
#include "ATen/ATen.h"

void cuda_forward_f16(int B, int T, int C, int H,
                      at::Half *r, at::Half *w, at::Half *k, at::Half *v, at::Half *a, at::Half *b,
                      at::Half *y, float *state_out, const float *state_in, int has_state_in);
void cuda_forward_bf16(int B, int T, int C, int H,
                       at::BFloat16 *r, at::BFloat16 *w, at::BFloat16 *k, at::BFloat16 *v,
                       at::BFloat16 *a, at::BFloat16 *b,
                       at::BFloat16 *y, float *state_out, const float *state_in, int has_state_in);

// inputs r,w,k,v,a,b : (B, T, C) contiguous, fp16 or bf16
// state_in (B, H, N, N) fp32 or empty tensor
// returns: y (B,T,C) same dtype, state_out (B,H,N,N) fp32
std::vector<torch::Tensor> forward(int64_t B, int64_t T, int64_t C, int64_t H,
                                   torch::Tensor r, torch::Tensor w, torch::Tensor k,
                                   torch::Tensor v, torch::Tensor a, torch::Tensor b,
                                   torch::Tensor state_in) {
    auto opts_y = r.options();
    torch::Tensor y = torch::empty({B, T, C}, opts_y);
    int has_state_in = (state_in.numel() > 0);
    torch::Tensor state_out = torch::empty({B, H, C / H, C / H},
        torch::TensorOptions().dtype(torch::kFloat32).device(r.device()));

    if (r.dtype() == torch::kHalf) {
        cuda_forward_f16((int)B, (int)T, (int)C, (int)H,
            r.data_ptr<at::Half>(), w.data_ptr<at::Half>(), k.data_ptr<at::Half>(),
            v.data_ptr<at::Half>(), a.data_ptr<at::Half>(), b.data_ptr<at::Half>(),
            y.data_ptr<at::Half>(),
            state_out.data_ptr<float>(),
            has_state_in ? state_in.data_ptr<float>() : nullptr,
            has_state_in);
    } else if (r.dtype() == torch::kBFloat16) {
        cuda_forward_bf16((int)B, (int)T, (int)C, (int)H,
            r.data_ptr<at::BFloat16>(), w.data_ptr<at::BFloat16>(), k.data_ptr<at::BFloat16>(),
            v.data_ptr<at::BFloat16>(), a.data_ptr<at::BFloat16>(), b.data_ptr<at::BFloat16>(),
            y.data_ptr<at::BFloat16>(),
            state_out.data_ptr<float>(),
            has_state_in ? state_in.data_ptr<float>() : nullptr,
            has_state_in);
    } else {
        TORCH_CHECK(false, "rwkv7 wkv kernel only supports fp16 / bf16, got ", r.dtype());
    }
    return {y, state_out};
}

TORCH_LIBRARY(rwkv7_cuda, m) {
    m.def("forward", forward);
}
