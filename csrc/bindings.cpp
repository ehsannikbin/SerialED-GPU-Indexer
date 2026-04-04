#include <torch/extension.h>
#include <tuple>

// Forward declarations of your functions
torch::Tensor compute_rotogram_cpp(
    torch::Tensor q_vecs, torch::Tensor q_norms,
    torch::Tensor rlps, torch::Tensor rlp_norms,
    int rot_size, float length_tol, float rot_scale, float rot_offset, int spin_steps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> refine_batch_cpp(
    torch::Tensor R_init_batch, torch::Tensor peaks_batch, torch::Tensor shifts_batch, 
    torch::Tensor masks_batch, torch::Tensor B_ref, torch::Tensor indices_ref,
    float k0, torch::Tensor k_in, float clen, float res,
    int steps, float def_penalty, float def_limit,
    float max_shift_limit, float shift_penalty_weight, float expanded_radius_multiplier,
    float lr_rot, float lr_shift, float lr_cell,
    float huber_delta, float radius_start, float radius_end);

// The SINGLE module definition for Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_rotogram_cpp", &compute_rotogram_cpp, "CUDA Rotogram Generation");
    m.def("refine_batch_cpp", &refine_batch_cpp, "Full C++ Refinement Loop");
}
