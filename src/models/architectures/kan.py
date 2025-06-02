import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    """
    KAN-layer using b-splines basis based on this paper:

    https://arxiv.org/abs/2404.19756

    This architecture fundamentally differs from MLPs by replacing fixed activation functions 
    with learnable univariate functions represented as B-spline bases 
    by decomposing multivariate functions into sums of univariate function.


    Using the model presented in the original paper turned out to be very impractical 
    due to the need to integrate many dependencies that contradict each other within the current project,
    so KAN-layer was rewritten from scratch with a focus on a more stable implementation on PyTorch with CUDA support:

    https://github.com/Blealtan/efficient-kan  

    Where:
        
    - B-spline parameterization enables continuous, piecewise-polynomial functions;
    - Grid-based routing follows the theoretical foundation of Kolmogorov's theorem;
    - Dual-path architecture (base + spline) enhances model expressivity;
    - Normalization of B-spline bases and grid perturbation thresholding (grid_eps) to prevent division-by-zero errors.
    """
    
    def __init__(
        self,
        in_features: int,                                       
        out_features: int,                                      
        grid_size: int = 5,                                     
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: nn.Module = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list = [-1, 1],
    ):
        super(KANLayer, self).__init__()
        self.in_features = in_features                                          # Size of input vector
        self.out_features = out_features                                        # Output vector size
        self.grid_size = grid_size                                              # Number of knot intervals for B-spline basis
        self.spline_order = spline_order                                        # Degree of B-spline polynomials (k=3: cubic splines)
        self.scale_noise = scale_noise                                          # Noise scaling for numerical stability during training
        self.scale_base = scale_base                                            # Linear transformation scaling
        self.scale_spline = scale_spline                                        # Spline path scaling
        self.enable_standalone_scale_spline = enable_standalone_scale_spline    # Optional standalone scaling mechanism for spline weights
        self.base_activation = base_activation()                                # Base activation function (SiLU chosen for its smoothness properties)
        self.grid_eps = grid_eps                                                # Grid perturbation threshold for numerical stability

        # B-spline Grid Construction
        h = (grid_range[1] - grid_range[0]) / grid_size
        # Grid: grid_size + 2 * spline_order + 1 point
        # Extended grid with boundary padding for B-spline continuity
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float) * h + grid_range[0]
        grid = grid.unsqueeze(0).expand(in_features, -1).contiguous()  # (in_features, grid_size + 2*spline_order + 1)
        self.register_buffer("grid", grid)

        # Linear transformation equivalent to traditional neural networks (Base Weight)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_bias = nn.Parameter(torch.Tensor(out_features))

        # Learnable B-spline coefficients (Spline Weight):
        # (out_features, in_features, grid_size + spline_order)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            # Initialize scale in 1.0
            self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features))

        # LayerNormalization for outputs
        self.norm_base = nn.LayerNorm(out_features)
        self.norm_spline = nn.LayerNorm(out_features)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Parameter initialization strategy combining:
        - base_weight: Kaiming initialization for base weights with SiLU gain adjustment;
        - base_bias: Zero initialization for biases;
        - spline_weight: Small random initialization for spline weights;
        - spline_scaler: ones (if it's standalone).
        """
        # Gain adjustment for SiLU activation
        gain = math.sqrt(2.0)  # SiLU ~ ReLU gain
        nn.init.kaiming_uniform_(self.base_weight, a=0, mode='fan_in', nonlinearity='relu')
        self.base_weight.data.mul_(self.scale_base * gain)
        nn.init.zeros_(self.base_bias)

        # Small random initialization for spline weights to break symmetry
        nn.init.uniform_(self.spline_weight, -self.scale_noise, self.scale_noise)

        if self.enable_standalone_scale_spline:
            # Identity initialization for spline scalers
            pass

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions using Cox-de Boor recursion.
        Normalization ensures partition-of-unity property for numerical stability
        Args:
            x: Input tensor of shape (N, in_features)
        Returns:
            bases: B-spline basis tensor of shape (N, in_features, grid_size + spline_order)
        """
        N = x.shape[0]
        # Expand grid to match input dimensions:
        # grid: (in_features, grid_points) -> (N, in_features, grid_points)
        grid = self.grid.unsqueeze(0).expand(N, -1, -1)  # (N, in_features, G)
        x_exp = x.unsqueeze(2)  # (N, in_features, 1)

         # Initial basis (zeroth order)
        bases = ((x_exp >= grid[:, :, :-1]) & (x_exp < grid[:, :, 1:])).to(x.dtype)  # (N, in_features, G-1)

        # Cox-de Boor Recursive B-spline construction
        for k in range(1, self.spline_order + 1):
            left_num = x_exp - grid[:, :, :-(k + 1)]
            left_den = grid[:, :, k:-1] - grid[:, :, :-(k + 1)] + 1e-8
            term1 = (left_num / left_den) * bases[:, :, :-1]

            right_num = grid[:, :, k + 1:] - x_exp
            right_den = grid[:, :, k + 1:] - grid[:, :, 1:-k] + 1e-8
            term2 = (right_num / right_den) * bases[:, :, 1:]
            bases = term1 + term2  # (N, in_features, grid_size + spline_order)

        # Normalize to maintain numerical stability = 1
        bases = bases / (bases.sum(dim=2, keepdim=True) + 1e-8)
        return bases.contiguous()  # (N, in_features, grid_size + spline_order)

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        """
        Apply scaling to spline weights if standalone scaling is enabled (adapting scaling mechnism)
        """
        if self.enable_standalone_scale_spline:
            # (out_features, in_features, grid_size + spline_order) *
            # (out_features, in_features, 1) -> (out_features, in_features, grid_size + spline_order)
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        else:
            return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining base and spline paths:
        - Base path: Linear transformation with SiLU activation
        - Spline path: B-spline basis expansion with learned coefficients
        - Path combination through summation
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)  # (N, in_features)

        # Base path introduces Standard linear transformation with activation
        base_act = self.base_activation(x_flat)  # (N, in_features)
        base_lin = F.linear(base_act, self.base_weight, self.base_bias)  # (N, out_features)
        base_out = self.norm_base(base_lin)  # (N, out_features)

        #  Spline path: B-spline basis expansion
        bspline = self.b_splines(x_flat)  # (N, in_features, grid_size + spline_order)
        bspline_flat = bspline.view(x_flat.size(0), -1)  # (N, in_features * (grid_size + spline_order))

        # Efficient weight application through linear operation:
        # (out_features, in_features, S) -> (out_features, in_features * S)
        spline_w_flat = self.scaled_spline_weight.view(self.out_features, -1)  # (out_features, in_features * S)
        spline_lin = F.linear(bspline_flat, spline_w_flat)  # (N, out_features)
        spline_out = self.norm_spline(spline_lin)  # (N, out_features)

        # Combine paths with residual connection-like behavior
        out = base_out + spline_out  # (N, out_features)
        out = out.view(*orig_shape[:-1], self.out_features)  # Restore original shape

        return out





