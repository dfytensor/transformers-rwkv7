// WKV-7 CUDA kernel — enhanced from the official BlinkDL/RWKV-LM wkv7.cu.
//
// Enhancements over upstream:
//   1. Templated on dtype F (Half = fp16, or BFloat16) so the model can run in either.
//   2. Writes the final recurrent state matrix S (B, H, N, N) fp32, so this kernel is a
//      drop-in for BOTH the parallel/training path and the RNN-decode (state-carry) path.
//
// Reference: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/cuda/wkv7.cu
#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _w,
                               const F *__restrict__ const _k, const F *__restrict__ const _v,
                               const F *__restrict__ const _a, const F *__restrict__ const _b,
                               F *__restrict__ const _y,
                               float *__restrict__ const _state_out,
                               const float *__restrict__ const _state_in,
                               const int has_state_in)
{
    const int e = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    float state[_N_] = {0};
    if (has_state_in) {
        // load initial state row i: state_in is (B, H, N, N), this thread owns row i, cols 0..N-1
        const int base = (e * H + h) * _N_ * _N_ + i * _N_;
        #pragma unroll
        for (int j = 0; j < _N_; j++) state[j] = _state_in[base + j];
    }
    __shared__ float r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];

    for (int _t = 0; _t < T; _t++)
    {
        const int t = e * T * C + h * _N_ + i + _t * C;
        __syncthreads();
        r[i] = float(_r[t]);
        w[i] = __expf(-__expf(float(_w[t])));
        k[i] = float(_k[t]);
        a[i] = float(_a[t]);
        b[i] = float(_b[t]);
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++) sa += a[j] * state[j];

        float vv = float(_v[t]);
        float y = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float &s = state[j];
            s = s * w[j] + k[j] * vv + sa * b[j];
            y += s * r[j];
        }
        _y[t] = F(y);
    }

    // write final state row i (cols 0..N-1)
    const int base = (e * H + h) * _N_ * _N_ + i * _N_;
    #pragma unroll
    for (int j = 0; j < _N_; j++) _state_out[base + j] = state[j];
}

// explicit instantiations for fp16 and bf16
void cuda_forward_f16(const int B, const int T, const int C, const int H,
                      at::Half *r, at::Half *w, at::Half *k, at::Half *v, at::Half *a, at::Half *b,
                      at::Half *y, float *state_out, const float *state_in, int has_state_in) {
    assert(H * _N_ == C);
    kernel_forward<at::Half><<<dim3(B * H), dim3(_N_)>>>(
        B, T, C, H, r, w, k, v, a, b, y, state_out, state_in, has_state_in);
}
void cuda_forward_bf16(const int B, const int T, const int C, const int H,
                       at::BFloat16 *r, at::BFloat16 *w, at::BFloat16 *k, at::BFloat16 *v,
                       at::BFloat16 *a, at::BFloat16 *b,
                       at::BFloat16 *y, float *state_out, const float *state_in, int has_state_in) {
    assert(H * _N_ == C);
    kernel_forward<at::BFloat16><<<dim3(B * H), dim3(_N_)>>>(
        B, T, C, H, r, w, k, v, a, b, y, state_out, state_in, has_state_in);
}
