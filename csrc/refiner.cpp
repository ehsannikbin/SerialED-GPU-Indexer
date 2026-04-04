#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <algorithm>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> refine_batch_cpp(
    torch::Tensor R_init_batch, torch::Tensor peaks_batch, torch::Tensor shifts_batch, 
    torch::Tensor masks_batch, torch::Tensor B_ref, torch::Tensor indices_ref,
    float k0, torch::Tensor k_in, float clen, float res,
    int steps, float def_penalty, float def_limit,
    float max_shift_limit, float shift_penalty_weight, float expanded_radius_multiplier,
    float lr_rot, float lr_shift, float lr_cell,    
    float huber_delta, float radius_start, float radius_end)
{

    pybind11::gil_scoped_release no_gil;
    int B = peaks_batch.size(0);
    auto device = peaks_batch.device();

    // 1. Initialize Batched Parameters
    auto R_curr = R_init_batch.clone().detach();
    auto shift_curr = shifts_batch.clone().detach().set_requires_grad(true);
    auto rot_delta = torch::zeros({B, 3}, torch::TensorOptions().device(device).requires_grad(true));
    auto B_delta = torch::zeros({B, 3, 3}, torch::TensorOptions().device(device).requires_grad(true));

    auto B_init_batch = B_ref.unsqueeze(0).expand({B, -1, -1});
    auto I_batch = torch::eye(3, torch::TensorOptions().device(device)).unsqueeze(0).expand({B, -1, -1});

    // 2. Setup Adam Optimizer (Custom learning rates per parameter)
    std::vector<torch::optim::OptimizerParamGroup> param_groups;
    
    auto opt_rot = std::make_unique<torch::optim::AdamOptions>(lr_rot);
    param_groups.emplace_back(std::vector<torch::Tensor>{rot_delta}, std::move(opt_rot));
    
    auto opt_shift = std::make_unique<torch::optim::AdamOptions>(lr_shift);
    param_groups.emplace_back(std::vector<torch::Tensor>{shift_curr}, std::move(opt_shift));
    
    auto opt_B = std::make_unique<torch::optim::AdamOptions>(lr_cell);
    param_groups.emplace_back(std::vector<torch::Tensor>{B_delta}, std::move(opt_B)); ;;
    
    torch::optim::Adam optimizer(param_groups);

    float start_rad = radius_start;
    float end_rad = radius_end;
    float max_shift = max_shift_limit / res;
    
    auto indices_ref_exp = indices_ref.unsqueeze(0).expand({B, -1, -1});

    // 3. The C++ Optimization Loop
    for (int i = 0; i < steps; ++i) {
        optimizer.zero_grad();
        float progress = (float)i / steps;
        float current_radius = start_rad * (1.0f - progress) + end_rad * progress;
        float basis_weight = progress < 0.3f ? 0.0f : 1.0f;

        // --- Batched Rodrigues Rotation ---
        auto angle = torch::norm(rot_delta, 2, 1);
        auto safe_angle = angle.clamp_min(1e-9);
        auto axis = rot_delta / safe_angle.unsqueeze(1);
        
        auto kx = axis.select(1, 0), ky = axis.select(1, 1), kz = axis.select(1, 2);
        auto z = torch::zeros_like(kx);
        
        auto K = torch::stack({z, -kz, ky, kz, z, -kx, -ky, kx, z}, 1).view({-1, 3, 3});
        auto s = torch::sin(angle).view({-1, 1, 1});
        auto c = torch::cos(angle).view({-1, 1, 1});
        
        auto R_delta_mat = I_batch + s * K + (1.0f - c) * torch::bmm(K, K);
        auto small_angle_mask = (angle < 1e-9).view({-1, 1, 1});
        R_delta_mat = torch::where(small_angle_mask, I_batch, R_delta_mat);
        
        auto R_total = torch::bmm(R_delta_mat, R_curr);

        // --- Batched Deformation ---
        auto effective_B_delta = B_delta * basis_weight * def_limit;
        auto B_new = torch::bmm(B_init_batch, I_batch + effective_B_delta);

        // --- Batched Observation Vectors ---
        auto px = peaks_batch.select(2, 0) + shift_curr.select(1, 0).unsqueeze(1);
        auto py = peaks_batch.select(2, 1) + shift_curr.select(1, 1).unsqueeze(1);
        auto pz = torch::full_like(px, clen);
        
        auto xyz_lab = torch::stack({px, py, pz}, -1);
        auto u_obs = torch::nn::functional::normalize(xyz_lab, torch::nn::functional::NormalizeFuncOptions().dim(-1));
        
        auto q_obs = k0 * u_obs - k_in.unsqueeze(0).unsqueeze(0);
        auto h_obs = torch::bmm(q_obs, R_total);

        // --- Batched Distances ---
        auto rlps_new = torch::bmm(indices_ref_exp, B_new.transpose(1, 2));
        
        auto dists = torch::cdist(h_obs, rlps_new);
        auto min_results = torch::min(dists, 2);
        auto min_dists = std::get<0>(min_results);
        auto idxs = std::get<1>(min_results);

        // --- Vectorized Match & Rejection ---
        auto idxs_exp = idxs.unsqueeze(-1).expand({-1, -1, 3});
        auto h_matched = torch::gather(rlps_new, 1, idxs_exp);
        
        auto q_pred = torch::bmm(h_matched, R_total.transpose(1, 2));
        auto dot = torch::sum(q_pred * u_obs, /*dim=*/2, /*keepdim=*/true);
        auto projection = dot * u_obs;
        auto rejection = q_pred - projection;
        auto err = torch::norm(rejection, 2, 2);
        
        auto loss_per_peak = torch::nn::functional::huber_loss(
            err, torch::zeros_like(err), 
            torch::nn::functional::HuberLossFuncOptions().reduction(torch::kNone).delta(huber_delta));

        // --- Strict Conditional Masking ---
        auto geom_mask = min_dists < current_radius;
        auto geom_mask_expanded = min_dists < (current_radius * expanded_radius_multiplier);
        
        auto valid_with_normal = geom_mask.logical_and(masks_batch).sum(1);
        auto use_expanded = valid_with_normal < 3;
        
        auto final_geom = torch::where(use_expanded.unsqueeze(1), geom_mask_expanded, geom_mask);
        auto final_mask = final_geom.logical_and(masks_batch);
        
        // --- Compute Batched Loss ---
        auto valid_counts_raw = final_mask.sum(1);
        auto valid_pattern_mask = (valid_counts_raw >= 3).to(torch::kFloat32);
        auto valid_counts = valid_counts_raw.clamp_min(1).to(torch::kFloat32);
        
        auto masked_loss = loss_per_peak * final_mask.to(torch::kFloat32);
        auto loss_fit_per_item = masked_loss.sum(1) / valid_counts;
        auto loss_fit = loss_fit_per_item.sum();
        
        auto shift_delta = shift_curr - shifts_batch;
        auto shift_sq = shift_delta.pow(2).sum(1);
        auto loss_shift = (shift_sq * valid_pattern_mask).sum() * shift_penalty_weight;
        
        auto B_sq = B_delta.pow(2).sum({1, 2});
        float reg_strength = 1.0f / std::max(def_penalty, 1.0f);
        auto loss_reg = (B_sq * valid_pattern_mask).sum() * reg_strength;
        
        auto loss = loss_fit + loss_shift + loss_reg;
        loss.backward();

        // --- HARD FREEZE for Invalid Patterns ---
        torch::Tensor saved_rot, saved_shift, saved_B;
        auto invalid_indices = (valid_pattern_mask == 0);
        
        {
            torch::NoGradGuard no_grad;
            if (invalid_indices.any().item<bool>()) {
                saved_rot = rot_delta.index({invalid_indices}).clone();
                saved_shift = shift_curr.index({invalid_indices}).clone();
                saved_B = B_delta.index({invalid_indices}).clone();
            }
        }

        optimizer.step();

        // Restore state for invalid patterns
        {
            torch::NoGradGuard no_grad;
            if (invalid_indices.any().item<bool>()) {
                rot_delta.index_put_({invalid_indices}, saved_rot);
                shift_curr.index_put_({invalid_indices}, saved_shift);
                B_delta.index_put_({invalid_indices}, saved_B);
            }
            auto min_bounds = shifts_batch - max_shift;
            auto max_bounds = shifts_batch + max_shift;
            shift_curr.copy_(torch::clamp(shift_curr, min_bounds, max_bounds));
            B_delta.clamp_(-1.0f, 1.0f);
        }
    }

    // 4. Final Parameter Extraction
    torch::Tensor R_final, B_final;
    {
        torch::NoGradGuard no_grad;
        auto angle = torch::norm(rot_delta, 2, 1);
        auto safe_angle = angle.clamp_min(1e-9);
        auto axis = rot_delta / safe_angle.unsqueeze(1);
        auto kx = axis.select(1, 0), ky = axis.select(1, 1), kz = axis.select(1, 2);
        auto z = torch::zeros_like(kx);
        auto K = torch::stack({z, -kz, ky, kz, z, -kx, -ky, kx, z}, 1).view({-1, 3, 3});
        auto s = torch::sin(angle).view({-1, 1, 1});
        auto c = torch::cos(angle).view({-1, 1, 1});
        auto R_delta_mat = I_batch + s * K + (1.0f - c) * torch::bmm(K, K);
        auto small_angle_mask = (angle < 1e-9).view({-1, 1, 1});
        R_delta_mat = torch::where(small_angle_mask, I_batch, R_delta_mat);
        
        R_final = torch::bmm(R_delta_mat, R_curr);
        B_final = torch::bmm(B_init_batch, I_batch + B_delta * def_limit);
    }

    return std::make_tuple(R_final, B_final, shift_curr.detach());
}

