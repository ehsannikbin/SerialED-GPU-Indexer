#include <torch/extension.h>

void rotogram_kernel_launcher(
    const float* q_vecs, const float* q_norms,
    const float* rlps, const float* rlp_norms,
    float* rotogram,
    int N, int M, int rot_size,
    float length_tol, float rot_scale, float rot_offset, int spin_steps);

torch::Tensor compute_rotogram_cpp(
    torch::Tensor q_vecs, torch::Tensor q_norms,
    torch::Tensor rlps, torch::Tensor rlp_norms,
    int rot_size, float length_tol, float rot_scale, float rot_offset, int spin_steps)
{
    TORCH_CHECK(q_vecs.is_cuda() && rlps.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(q_vecs.is_contiguous() && rlps.is_contiguous(), "Inputs must be contiguous");

    int N = q_vecs.size(0);
    int M = rlps.size(0);

    // Initialize 3D Rotogram with zeros
    auto rotogram = torch::zeros({rot_size, rot_size, rot_size}, 
        torch::TensorOptions().device(q_vecs.device()).dtype(torch::kFloat32));

    if (N == 0 || M == 0) return rotogram;

    rotogram_kernel_launcher(
        q_vecs.data_ptr<float>(),
        q_norms.data_ptr<float>(),
        rlps.data_ptr<float>(),
        rlp_norms.data_ptr<float>(),
        rotogram.data_ptr<float>(),
        N, M, rot_size, length_tol, rot_scale, rot_offset, spin_steps
    );

    return rotogram;
}

