import argparse
import torch
import os

class PinkIndexerConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PyTorch PinkIndexer for Electron Diffraction")
        
        # --- File I/O ---
        self.parser.add_argument("--wdir", type=str, default=".", help="Base working directory.")
        self.parser.add_argument("--geometry", type=str, required=True, help="Path to .geom file")
        self.parser.add_argument("--cell", type=str, required=True, help="Path to .cell file")
        self.parser.add_argument("--input", type=str, required=True, help="Path to .lst file")
        self.parser.add_argument("-o", "--output", type=str, default="pinkIndexer.stream", help="Output stream filename")
        self.parser.add_argument("--pixel-convention", type=str, default="corner", choices=["corner", "center"],
                                 help="Pixel coordinate convention: 'corner' (0,0 is edge) or 'center' (0,0 is center). Default 'corner'.")
        
        # --- Hardware ---
        self.parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
        self.parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
        self.parser.add_argument("--workers-per-gpu", type=int, default=2, 
                                 help="Number of worker processes to launch per GPU to hide latency. Default 4.")

        # --- Performance ---
        self.parser.add_argument("--gpu-batch-size", type=int, default=8, 
                                 help="Number of patterns to process in parallel on the GPU. Default 32.")
        #self.parser.add_argument("--batch-size-refinement", type=int, default=32, help="Refinement seeds per pattern. Default 32.")
        #self.parser.add_argument("--chunk-size-rotogram", type=int, default=4, help="Rotogram chunk size. Default 4.")
        #self.parser.add_argument("--chunk-size-cdist", type=int, default=20000, help="Matching chunk size. Default 100000.")
        self.parser.add_argument("--chunk-size-events", type=int, default=2000, 
                                 help="Max events per chunk loaded into RAM to prevent OOM. Default 2000.")
        
        # --- Refinement Hyperparameters ---
        self.parser.add_argument("--lr-rot", type=float, default=0.005, help="Learning rate for rotation adjustment.")
        self.parser.add_argument("--lr-shift", type=float, default=0.0001, help="Learning rate for detector center shift.")
        self.parser.add_argument("--lr-cell", type=float, default=0.008, help="Learning rate for unit cell deformation.")
        self.parser.add_argument("--huber-delta", type=float, default=0.003, help="Delta threshold for Huber loss.")
        self.parser.add_argument("--radius-start", type=float, default=0.05, help="Starting funnel radius in 1/A.")
        self.parser.add_argument("--radius-end", type=float, default=0.003, help="Ending funnel radius in 1/A.")

        # --- PinkIndexer Algorithm ---
        self.parser.add_argument("--pinkIndexer-considered-peaks-count", type=int, default=3, help="Peak count level (0-4).")
        self.parser.add_argument("--rotogram-peaks", type=int, default=50, help="Maximum number of bright peaks used to generate the initial rotogram.")
        self.parser.add_argument("--pinkIndexer-angle-resolution", type=int, default=4, help="Angle resolution level (0-4).")
        #self.parser.add_argument("--pinkIndexer-refinement-type", type=int, default=5, help="Refinement mode.")
        self.parser.add_argument("--pinkIndexer-refinement-steps", type=int, default=100, help="Total refinement steps (split between rigid/deform).")
        self.parser.add_argument("--pinkIndexer-tolerance", type=float, default=0.06, help="Indexing tolerance.")
        self.parser.add_argument("--res-limit-tolerance", type=float, default=0.5, 
                                 help="Relative fractional tolerance (in RLU) used strictly for calculating the diffraction resolution limit.")
        self.parser.add_argument("--pinkIndexer-reflection-radius", type=float, default=0.002, help="Reflection radius (1/A).")
        self.parser.add_argument("--pinkIndexer-max-resolution-for-indexing", type=float, default=0.7, help="Max resolution (A).")


        self.parser.add_argument("--max-shift-limit", type=float, default=4.0, help="Maximum allowed detector shift in pixels.")
        self.parser.add_argument("--shift-penalty-weight", type=float, default=10.0, help="Regularization weight preventing extreme detector shifts.")
        self.parser.add_argument("--rotogram-shell-multiplier", type=float, default=5.0, help="Multiplier for shell thickness in rotogram generation.")
        self.parser.add_argument("--expanded-radius-multiplier", type=float, default=2.0, help="Multiplier for search radius if initial match < 3 peaks.")
        self.parser.add_argument("--rotogram-spin-steps", type=int, default=180, help="Number of spin angles evaluated per peak in the rotogram (max 1024).")

        # --- Unit Cell Constraints ---
        self.parser.add_argument("--deformation-penalty", type=float, default=4000.0,
                                 help="Strength of the constraint to keep refined cell close to input cell.")
        
        # --- Rejection & Limits ---
        self.parser.add_argument("--min-n-peaks", type=int, default=15, 
                                 help="Minimum number of peaks required to attempt indexing.")
        
        self.parser.add_argument("--deformation-limit-percent", type=float, default=0.05,
                                 help="Maximum allowed cell deformation (fraction). Used for both refinement clamp and rejection.")
        
        self.parser.add_argument("--rejection-iwr-threshold", type=float, default=0.1,
                                 help="Intensity Weighted Recall threshold (0.0-1.0). Reject if explained intensity < threshold.")
        
        self.parser.add_argument("--rejection-rmsd-factor", type=float, default=0.4,
                                 help="RMSD rejection factor. Reject if RMSD > (tolerance * factor).")

        # --- Integration ---
        self.parser.add_argument("--int-radii", type=str, default="3,5,6", 
                                 help="Integration radii (inner, mid, outer). Default '4,5,7'")
        
        self.parser.add_argument("--push-res", type=float, default=0.4,
                                 help="Push the resolution limit for integration by this amount (A).")

        self.args = None

    def parse(self):
        self.args = self.parser.parse_args()
        
        # Parse radii string to list of ints
        try:
            self.args.int_radii = [int(x) for x in self.args.int_radii.split(',')]
            if len(self.args.int_radii) != 3: raise ValueError
        except:
            print("Error: --int-radii must be 3 integers separated by commas (e.g. 4,5,7)")
            exit(1)

        if self.args.device == 'cuda' and not torch.cuda.is_available():
            self.args.device = 'cpu'
        if not os.path.exists(self.args.wdir):
            print(f"Error: Working directory '{self.args.wdir}' does not exist.")
            exit(1)
        return self.args
    