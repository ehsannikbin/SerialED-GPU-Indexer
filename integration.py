import torch
import numpy as np
import math

class Integrator:
    def __init__(self, geom_params, radii=(4, 5, 7), device='cuda', pixel_convention='corner'):
        """
        Mimics CrystFEL integration.c (Ring Integration).
        radii: (r_inner, r_mid, r_outer)
           r < r_inner: Peak
           r_inner <= r < r_mid: Buffer (Ignore)
           r_mid <= r < r_outer: Background
        """
        self.device = device
        self.pixel_convention = pixel_convention
        self.r_inner = radii[0]
        self.r_mid = radii[1]
        self.r_outer = radii[2]
        self.geom = geom_params
        
        # Pre-compute box size based on outer radius
        self.box_size = int(math.ceil(self.r_outer)) * 2 + 1
        self.half_w = self.box_size // 2

        # Create localized mask template
        self.mask_template = self._create_ring_mask()

    def _create_ring_mask(self):
        """
        Creates a 2D mask template:
        0 = Ignore/Buffer
        1 = Background
        2 = Peak
        """
        y, x = torch.meshgrid(
            torch.arange(-self.half_w, self.half_w + 1, device=self.device),
            torch.arange(-self.half_w, self.half_w + 1, device=self.device),
            indexing='ij'
        )
        r2 = x**2 + y**2
        
        mask = torch.zeros_like(r2, dtype=torch.int8)
        
        # Background: r_mid^2 <= r^2 < r_outer^2
        # mask[(r2 >= self.r_mid**2) & (r2 < self.r_outer**2)] = 1
        
        # Peak: r^2 < r_inner^2
        # mask[r2 < self.r_inner**2] = 2

        # Background: r_mid^2 <= r^2 <= r_outer^2
        mask[(r2 >= self.r_mid**2) & (r2 <= self.r_outer**2)] = 1
        
        # Peak: r^2 <= r_inner^2
        mask[r2 <= self.r_inner**2] = 2
        
        return mask

    def integrate(self, image_tensor, pred_coords):
        """
        image_tensor: (H, W) tensor on device
        pred_coords: (N, 2) tensor of predicted (fs, ss) locations
        
        Returns:
            intensities: (N,)
            sigmas: (N,)
            peak_vals: (N,) max value in peak
            bg_means: (N,)
        """
        if len(pred_coords) == 0:
            return (torch.tensor([], device=self.device), 
                    torch.tensor([], device=self.device),
                    torch.tensor([], device=self.device),
                    torch.tensor([], device=self.device))

        H, W = image_tensor.shape
        N = pred_coords.shape[0]
        
        # Output containers
        out_I = torch.zeros(N, device=self.device)
        out_sigma = torch.zeros(N, device=self.device)
        out_peak = torch.zeros(N, device=self.device)
        out_bg = torch.zeros(N, device=self.device)
        
        # Round predictions to nearest integer pixel for box centering
        # centers = torch.round(pred_coords).long()

        if self.pixel_convention == 'corner':
            centers = torch.floor(pred_coords).long()   # Floor predictions to nearest integer pixel for box corner following CrystFEL's convention 
        else:
            centers = torch.round(pred_coords).long()
        
        # Boundary checks
        valid_mask = (centers[:, 0] >= self.half_w) & (centers[:, 0] < W - self.half_w) & \
                     (centers[:, 1] >= self.half_w) & (centers[:, 1] < H - self.half_w)
        
        # Process only valid reflections inside image bounds
        valid_indices = torch.nonzero(valid_mask).squeeze()
        if valid_indices.numel() == 0:
             return out_I, out_sigma, out_peak, out_bg
             
        if valid_indices.dim() == 0: valid_indices = valid_indices.unsqueeze(0)

        # --- Batch Extraction (Optional optimization: use unfold) ---
        # For simplicity and clarity mimicking the C code, we iterate or use simplified batching.
        # Given the number of reflections (~100-2000), a loop with tensor ops is acceptable,
        # but Unfold is faster. Let's use loop for readability matching logic unless N is huge.
        
        # Note: In PyTorch (y, x) indexing vs (fs, ss). 
        # Usually fs=x (col), ss=y (row). 
        # image[y, x] => image[ss, fs]
        
        for i in valid_indices:
            cx, cy = centers[i, 0], centers[i, 1] # fs, ss
            
            # Slice: [row_start:row_end, col_start:col_end] -> [ss, fs]
            patch = image_tensor[cy - self.half_w : cy + self.half_w + 1,
                                 cx - self.half_w : cx + self.half_w + 1]
            
            # 1. Identify Regions
            mask_pk = (self.mask_template == 2)
            mask_bg = (self.mask_template == 1)
            
            n_pk = mask_pk.sum().float()
            n_bg = mask_bg.sum().float()
            
            if n_pk < 4 or n_bg < 4: continue

            # 2. Background Statistics
            bg_pixels = patch[mask_bg].float()
            bg_mean = torch.mean(bg_pixels)
            bg_var = torch.var(bg_pixels, unbiased=False) # Sample variance
            
            # 3. Peak Summation
            pk_pixels = patch[mask_pk].float()
            sum_pk = torch.sum(pk_pixels)
            
            # 4. Intensity Calculation
            # I = Sum_pk - (N_pk * Mean_bg)
            intensity = sum_pk - (n_pk * bg_mean)
            
            # 5. Sigma (Error) Calculation
            # Mimicking integration.c: sigma = sqrt(sig2_poisson + n_peak * sig2_bg)
            # where sig2_poisson approx Intensity (or 0 if negative)
            
            # ADU per photon (gain). If unknown, assume 1.0.
            adu_per_photon = 1.0 
            
            # Poisson variance part
            # If I > 0, var = I. If I < 0 (dip), var ~ |I| (approx)
            # The C code handles negative intensity logic specifically, 
            # effectively treating it as Poissonian noise of the background level 
            # if it dips too low, but standard approx is abs(I) + bg noise.
            
            # sig2_poisson = torch.abs(intensity) 
            sig2_poisson = torch.clamp(torch.abs(intensity), min=1.0)
            
            # Full variance
            sigma = torch.sqrt(sig2_poisson + (n_pk * bg_var))
            
            out_I[i] = intensity
            out_sigma[i] = sigma
            out_peak[i] = torch.max(pk_pixels) if len(pk_pixels) > 0 else 0
            out_bg[i] = bg_mean

        return out_I, out_sigma, out_peak, out_bg
    
    def integrate_batch(self, images_batch, centers_batch, masks_batch):
        """
        Batched integration.
        images_batch: (B, H, W)
        centers_batch: (B, N, 2)  [x, y]
        masks_batch: (B, N)      Boolean mask for valid peaks
        
        Returns: I, sigma, background, peak_max (all BxN)
        """
        B, N, _ = centers_batch.shape
        H, W = images_batch.shape[1], images_batch.shape[2]
        
        # 1. Round centers to nearest integer (Standard CrystFEL-like integration)
        # centers are (x, y). We need (row, col) -> (y, x)
        # cx = torch.round(centers_batch[:, :, 0]).long()
        # cy = torch.round(centers_batch[:, :, 1]).long()

        if self.pixel_convention == 'corner':
            cx = torch.floor(centers_batch[:, :, 0]).long()
            cy = torch.floor(centers_batch[:, :, 1]).long()
        else:
            cx = torch.round(centers_batch[:, :, 0]).long()
            cy = torch.round(centers_batch[:, :, 1]).long()
        
        # 2. Generate patch offsets
        # self.mask_template is (Box, Box). We need offsets relative to center.
        # Create grid of offsets (dy, dx)
        r = self.half_w
        dy = torch.arange(-r, r + 1, device=self.device)
        dx = torch.arange(-r, r + 1, device=self.device)
        grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')
        
        # Flatten grid: (K*K)
        off_y = grid_y.flatten()
        off_x = grid_x.flatten()
        n_pixels = off_y.shape[0]
        
        # 3. Compute absolute pixel coordinates for all peaks in batch
        # (B, N, 1) + (K*K) -> (B, N, K*K)
        patch_y = cy.unsqueeze(2) + off_y.unsqueeze(0).unsqueeze(0)
        patch_x = cx.unsqueeze(2) + off_x.unsqueeze(0).unsqueeze(0)
        
        # 4. Handle Bounds (Clamp to edges to avoid crash, we will mask later)
        patch_y_clamped = patch_y.clamp(0, H - 1)
        patch_x_clamped = patch_x.clamp(0, W - 1)
        
        # 5. Gather Pixels
        # Linear index for gather: b * (H*W) + y * W + x
        batch_idx = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, N, n_pixels)
        flat_indices = batch_idx * (H * W) + patch_y_clamped * W + patch_x_clamped
        
        # Flatten images to (B*H*W) for gather
        images_flat = images_batch.view(-1)
        # patches: (B, N, K*K)
        patches = torch.gather(images_flat, 0, flat_indices.view(-1)).view(B, N, n_pixels)
        
        # 6. Apply Integration Mask
        # mask_template: 0=Ignore, 1=Bg, 2=Peak
        mask_flat = self.mask_template.view(-1).unsqueeze(0).unsqueeze(0) # (1, 1, K*K)
        
        is_bg = (mask_flat == 1)
        is_pk = (mask_flat == 2)
        
        # 7. Compute Statistics (Vectorized)
        # Background
        bg_vals = patches * is_bg.float()
        # Count valid bg pixels (masking invalid peaks or out of bounds could happen here)
        n_bg = is_bg.sum()
        
        bg_sum = bg_vals.sum(dim=2)
        bg_mean = bg_sum / (n_bg + 1e-9)
        
        # Variance calculation for Bg (Sum of squares trick or explicit)
        # Var = E[X^2] - (E[X])^2
        bg_sq_sum = (bg_vals ** 2).sum(dim=2)
        bg_var = (bg_sq_sum / (n_bg + 1e-9)) - (bg_mean ** 2)
        
        # Peak
        pk_vals = patches * is_pk.float()
        sum_pk = pk_vals.sum(dim=2)
        n_pk = is_pk.sum()
        
        # Intensity
        intensity = sum_pk - (n_pk * bg_mean)
        
        # Sigma
        # sig2_poisson = torch.abs(intensity)
        sig2_poisson = torch.abs(intensity).clamp(min=1.0)
        sigma = torch.sqrt(sig2_poisson + n_pk * bg_var)
        
        peak_max = pk_vals.max(dim=2)[0]
        
        # Zero out results for invalid peaks (from padding)
        valid = masks_batch.float()
        return intensity * valid, sigma * valid, bg_mean * valid, peak_max * valid
    