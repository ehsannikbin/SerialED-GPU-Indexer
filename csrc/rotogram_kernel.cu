#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

__global__ void rotogram_kernel(
    const float* __restrict__ q_vecs,
    const float* __restrict__ q_norms,
    const float* __restrict__ rlps,
    const float* __restrict__ rlp_norms,
    float* __restrict__ rotogram,
    int N, int M, int rot_size,
    float length_tol, float rot_scale, float rot_offset, int spin_steps)
{
    // n: Peak index, m: Reference RLP index
    int m = blockIdx.x; 
    int n = blockIdx.y; 
    int spin_idx = threadIdx.x; // 0 to 179

    if (n >= N || m >= M || spin_idx >= spin_steps) return;

    float q_len = q_norms[n];
    float h_len = rlp_norms[m];

    // Shell thickness check: Exit the entire warp immediately if out of bounds
    if (fabsf(h_len - q_len) >= length_tol) return;

    // Load coordinates
    float qx = q_vecs[n*3 + 0], qy = q_vecs[n*3 + 1], qz = q_vecs[n*3 + 2];
    float hx = rlps[m*3 + 0],   hy = rlps[m*3 + 1],   hz = rlps[m*3 + 2];

    // Normalize vectors
    float inv_q = 1.0f / q_len;
    qx *= inv_q; qy *= inv_q; qz *= inv_q;

    float inv_h = 1.0f / h_len;
    hx *= inv_h; hy *= inv_h; hz *= inv_h;

    // Cross product: h_dir x q_dir
    float cx = hy * qz - hz * qy;
    float cy = hz * qx - hx * qz;
    float cz = hx * qy - hy * qx;

    float sin_theta = sqrtf(cx*cx + cy*cy + cz*cz);
    if (sin_theta <= 1e-6f) return;

    float cos_theta = hx*qx + hy*qy + hz*qz;
    float theta = atan2f(sin_theta, cos_theta);

    // Normalize cross axis
    float inv_sin = 1.0f / sin_theta;
    cx *= inv_sin; cy *= inv_sin; cz *= inv_sin;

    // Build R0 (Rotation around cross_axis by theta)
    float s0 = sinf(theta), c0 = cosf(theta), omc0 = 1.0f - c0;
    float R0[3][3] = {
        {c0 + cx*cx*omc0,       cx*cy*omc0 - cz*s0, cx*cz*omc0 + cy*s0},
        {cy*cx*omc0 + cz*s0,    c0 + cy*cy*omc0,    cy*cz*omc0 - cx*s0},
        {cz*cx*omc0 - cy*s0,    cz*cy*omc0 + cx*s0, c0 + cz*cz*omc0}
    };

    // Build R_spin (Rotation around q_dir by spin_angle)
    // Note: PyTorch linspace(0, 2pi, 180) divides by 179
    float spin_angle = spin_idx * (2.0f * CUDART_PI_F / (float)(spin_steps - 1));
    float s_spin = sinf(spin_angle), c_spin = cosf(spin_angle), omc_spin = 1.0f - c_spin;
    float Rs[3][3] = {
        {c_spin + qx*qx*omc_spin,    qx*qy*omc_spin - qz*s_spin, qx*qz*omc_spin + qy*s_spin},
        {qy*qx*omc_spin + qz*s_spin, c_spin + qy*qy*omc_spin,    qy*qz*omc_spin - qx*s_spin},
        {qz*qx*omc_spin - qy*s_spin, qz*qy*omc_spin + qx*s_spin, c_spin + qz*qz*omc_spin}
    };

    // R_tot = Rs * R0
    float Rt[3][3];
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            Rt[i][j] = Rs[i][0]*R0[0][j] + Rs[i][1]*R0[1][j] + Rs[i][2]*R0[2][j];
        }
    }

    // Extract Angle-Axis from R_tot
    float tr = Rt[0][0] + Rt[1][1] + Rt[2][2];
    float cos_t = fmaxf(-1.0f, fminf(1.0f, (tr - 1.0f) * 0.5f));
    float theta_tot = acosf(cos_t);

    float ax = Rt[2][1] - Rt[1][2];
    float ay = Rt[0][2] - Rt[2][0];
    float az = Rt[1][0] - Rt[0][1];

    float sin_t = sinf(theta_tot);

    // Standard case
    if (sin_t > 1e-4f) {
        float inv_2sin = 1.0f / (2.0f * sin_t);
        ax *= inv_2sin; 
        ay *= inv_2sin; 
        az *= inv_2sin;
    } 
    // Singularity case: Theta is near 180 degrees (cos_t is near -1)
    else if (cos_t < -0.99f) {
        float xx = (Rt[0][0] + 1.0f) * 0.5f;
        float yy = (Rt[1][1] + 1.0f) * 0.5f;
        float zz = (Rt[2][2] + 1.0f) * 0.5f;
        float xy = (Rt[0][1] + Rt[1][0]) * 0.25f;
        float xz = (Rt[0][2] + Rt[2][0]) * 0.25f;
        float yz = (Rt[1][2] + Rt[2][1]) * 0.25f;

        // Find the largest diagonal element to avoid numerical instability
        if (xx > yy && xx > zz) {
            ax = sqrtf(fmaxf(xx, 0.0f));
            ay = xy / ax;
            az = xz / ax;
        } else if (yy > zz) {
            ay = sqrtf(fmaxf(yy, 0.0f));
            ax = xy / ay;
            az = yz / ay;
        } else {
            az = sqrtf(fmaxf(zz, 0.0f));
            ax = xz / az;
            ay = yz / az;
        }
        
        // Normalize
        float len = sqrtf(ax*ax + ay*ay + az*az);
        if(len > 1e-6f) { 
            ax /= len; 
            ay /= len; 
            az /= len; 
        }
    }

    // Convert to v_map vector
    float mag = atanf(theta_tot / 4.0f);
    float vx = mag * ax;
    float vy = mag * ay;
    float vz = mag * az;

    // Convert to grid coordinates
    int gx = __float2int_rd(vx * rot_scale + rot_offset);
    int gy = __float2int_rd(vy * rot_scale + rot_offset);
    int gz = __float2int_rd(vz * rot_scale + rot_offset);

    // Bounds check and Atomic Add
    if (gx >= 0 && gx < rot_size && gy >= 0 && gy < rot_size && gz >= 0 && gz < rot_size) {
        int idx = gx * rot_size * rot_size + gy * rot_size + gz;
        atomicAdd(&rotogram[idx], 1.0f);
    }
}

// C++ Launcher function
void rotogram_kernel_launcher(
    const float* q_vecs, const float* q_norms,
    const float* rlps, const float* rlp_norms,
    float* rotogram,
    int N, int M, int rot_size,
    float length_tol, float rot_scale, float rot_offset, int spin_steps)
{
    dim3 blocks(M, N);
    dim3 threads(spin_steps); // 180 threads per block exactly match the 180 spin steps
    
    rotogram_kernel<<<blocks, threads>>>(
        q_vecs, q_norms, rlps, rlp_norms, rotogram,
        N, M, rot_size, length_tol, rot_scale, rot_offset, spin_steps
    );
}
