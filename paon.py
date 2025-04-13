import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm
from torchvision.ops import DeformConv2d

# ==========================================================================================================================
#region Shifter
# ==========================================================================================================================

class CustomAdaptiveAverage(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=(0, 2, 3), keepdim=True)
    
# __________________________________________________________________________________________________________________________
#
class Viewer(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        return x.view(*self.dim)
    
# __________________________________________________________________________________________________________________________
#
class Shifter(nn.Module):
    r"""
    channel: [integer]
        The number of channels of the features to be shifted.
    bias_range: [integer]
        The range in which the optimal kernel shift is evaluated.
    shift_is_tensor: [bool]
        If False, the shifts are definitely learned through a small network; otherwise it depends on the bias_range being
        equal to 0 (False) or bigger than 0 (True).
    bias_learn: [bool]
        True for learnable, False for constant shifts.
    bias_round: [bool]
        If True, then the center shifts will be rounded to the nearest integer.
    shift_padding_mode: [string]
        Padding mode for the shifter. It must be one of "zeros", "border", "reflection".
    """
    def __init__(
        self, channel: int, bias_range: int, shift_is_tensor: bool, bias_learn: bool, bias_round: bool, 
        shift_padding_mode: str
    ):
        super().__init__()

        if bias_range == 0 or (bias_range > 0 and not shift_is_tensor):
            self.shift_is_tensor = False
            self.center_bias = nn.Sequential(
                CustomAdaptiveAverage(),
                nn.Conv2d(channel, 2*channel, 1),
                nn.PReLU(2*channel),
                Viewer((channel, 2, 1))
            )
        else:
            self.shift_is_tensor = True
            self.bias_learn = bias_learn

            # Every neuron has in_chx2 many different bias pairs
            if self.bias_learn:
                self.center_bias = nn.Parameter(torch.Tensor(channel, 2))
            else:
                self.register_buffer("center_bias", torch.Tensor(channel, 2))

        self.ch = channel
        self.br = bias_range
        self.bias_round = bias_round
        self.spm = shift_padding_mode
        
        self.reset_parameters()

    # ----------------------------------------------------------------------------------------------------------------------
    def reset_parameters(self):
        # If the bias range is greater than 0, then we can have the bias parameter
        if self.shift_is_tensor:
            nn.init.uniform_(self.center_bias, a=-self.br, b=self.br)
        
            if self.bias_learn and self.bias_round:
                with torch.no_grad(): self.center_bias.data.round_()
        # If the bias range is 0, then we have a bias learner network
        else:
            nn.init.constant_(self.center_bias[1].weight, 0.)
            nn.init.constant_(self.center_bias[1].bias, 0.)

    # ----------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        # First, take the center biases (cb).
        if self.shift_is_tensor:
            cb = self.center_bias

            # Clamp the center bias in case of too much shift
            if self.bias_learn:
                cb = torch.clamp(cb, min=-self.br, max=self.br)
                # cb = torch.where(torch.abs(cb) >= self.br, self.br*torch.tanh(cb / self.br), cb)
        else:
            cb = self.center_bias(x).view(self.ch, -1)

            # # Clamp the center bias in case of too much shift
            # if self.br > 0:
            #     cb = torch.where(torch.abs(cb) >= self.br, self.br*torch.tanh(cb / self.br), cb)

        # Round the biases to the integer values
        if self.bias_round: cb = torch.round(cb)

        # Rearrange the input so that every channel can be shifted
        x = x.permute(1, 0, 2, 3)

        # Take the shape of the input
        c, _, h, w = x.size()

        # Normalize the coordinates to [-1, 1] range which is necessary for the grid
        a_r = cb[:,:1] / (w/2)
        b_r = cb[:,1:] / (h/2)

        # Create the transformation matrix
        aff_mtx = torch.eye(3).to(x.device)
        aff_mtx = aff_mtx.repeat(c, 1, 1)
        aff_mtx[..., 0, 2:3] += a_r
        aff_mtx[..., 1, 2:3] += b_r

        # Create the new grid
        grid = F.affine_grid(aff_mtx[..., :2, :3], x.size(), align_corners=False)

        # Interpolate the input values
        x = F.grid_sample(x, grid, mode="bilinear", align_corners=False, padding_mode=self.spm)

        # Rearrange the input to its original shape
        x = x.permute(1, 0, 2, 3)
        return x
#endregion
    
# ==========================================================================================================================
#region Padé Approximant Layer Base Class
# ==========================================================================================================================

class PaLaBase(nn.Module):
    r"""
    in_ch: [integer]
        Number of input channels. 
    out_ch: [integer]
        Number of output channels. 
    degrees: Tuple[integer]
        Truncation degrees of the Padé layer for numerator and denominator, respectively.
    paon_type: [string]
        The Paon type to be used in the layer. "a" for Paon-A (absolute), "s" for Paon-S (smooth).
    """
    def __init__(
        self, in_ch: int, out_ch: int, degrees: tuple[int, int], paon_type: str
    ):
        super().__init__()

        # Degree check
        # =========================================
        assert len(degrees) == 2, \
            f"As a ratio of polynomials, Padé approximant exactly needs two degrees but got {len(degrees)}."
        assert degrees != (0,0), "Both of the degrees cannot be 0 at the same time."
        assert degrees[0] >= 0 and degrees[1] >= 0, "None of the degrees can be negative."

        # Attributes
        # =========================================
        self.in_ch     = in_ch
        self.out_ch    = out_ch
        self.degrees   = degrees
        self.paon_type = paon_type
        
        # Parameters
        # =========================================
        # Get weights
        self.get_m_poly = None
        self.get_n_poly = None
        
        # w0 term
        self.w0 = nn.Parameter(torch.Tensor(out_ch))
        
        # Paon type
        # =========================================
        if paon_type == "a":
            self.prepare_for_div = self.prepare_for_div_a
            self.after_poly      = self.after_poly_a
        elif paon_type == "s":
            self.prepare_for_div = self.prepare_for_div_s
            self.after_poly      = self.after_poly_s
        else:
            raise ValueError(f"Unsupported Paon type {paon_type}.")

    # ______________________________________________________________________________________________________________________
    # 
    def prepare_polynomial(self, x, n, get_poly, offset=None):
        pass

    # ______________________________________________________________________________________________________________________
    # 
    def after_poly_a(self, x, n):
        # If it is the denomanator convolution, then it should be absolute.
        if self.degrees[n] == 1: 
            return torch.abs(x) if n == 1 else x
        
        # Sum the different degrees
        o = self.degrees[n]
        ch = self.out_ch
        new_order = [(k * ch) % (ch * o) + k // o for k in range(ch * o)]
        x = x[:, new_order]
        b, rest = x.shape[0], x.shape[2:]
        x = x.view(b, ch, o, *rest)
        return torch.abs(x).sum(dim=2) if n == 1 else x.sum(dim=2)

    # ______________________________________________________________________________________________________________________
    # 
    def after_poly_s(self, x, n):
        # If the degree is 1, then return the tensor directly
        if self.degrees[n] == 1: 
            return x

        # Sum the different degrees
        o = self.degrees[n]
        ch = self.out_ch
        new_order = [(k * ch) % (ch * o) + k // o for k in range(ch * o)]
        x = x[:, new_order]
        b, rest = x.shape[0], x.shape[2:]
        x = x.view(b, ch, o, *rest)
        return x.sum(dim=2), x[:, :, :o-1].sum(dim=2)
    
    # ______________________________________________________________________________________________________________________
    # 
    def prepare_for_div_a(self, pm_tuple, pm_bias, qn_tuple, degrees):
        view_dims = (1, 1, -1) if qn_tuple.ndim == 3 else (1, -1, 1, 1)
        if pm_tuple is None:
            return pm_bias.view(*view_dims), torch.ones_like(qn_tuple) + qn_tuple
        return pm_tuple + pm_bias.view(*view_dims), torch.ones_like(qn_tuple) + qn_tuple
        
    # ______________________________________________________________________________________________________________________
    # 
    def prepare_for_div_s(self, pm_tuple, pm_bias, qn_tuple, degrees):
        # We need q_N and q_(N-1) polynomials
        if degrees[1] == 1:
            qN = qn_tuple + torch.ones_like(qn_tuple)

            # If the first degree is 0, then we have a Padé approximant depending only on the denominator polynomial. Since
            # here the bias term serves as the multiplier for each channel for qN, we can discard it assuming that if any 
            # amplification is necessary, the gradient descent should modify the weights accordingly.
            if degrees[0] == 0:
                view_dims = (1, 1, -1) if qN.ndim == 3 else (1, -1, 1, 1)
                return qN*pm_bias.view(*view_dims), qN**2 + torch.ones_like(qN) # qN*pm_bias, ...
            
            if degrees[0] == 1:
                view_dims = (1, 1, -1) if pm_tuple.ndim == 3 else (1, -1, 1, 1)
                pM = pm_tuple + pm_bias.view(*view_dims)
                return qN*pM + pm_bias.view(*view_dims), qN**2 + torch.ones_like(qN)
            
            view_dims = (1, 1, -1) if pm_tuple[0].ndim == 3 else (1, -1, 1, 1)
            pM = pm_tuple[0] + pm_bias.view(*view_dims)
            pm = pm_tuple[1] + pm_bias.view(*view_dims)
            return qN*pM + pm, qN**2 + torch.ones_like(qN)
        else:
            qN = qn_tuple[0] + torch.ones_like(qn_tuple[0])
            qn = qn_tuple[1] + torch.ones_like(qn_tuple[1])

            # If the first degree is 0, then we have a Padé approximant depending only on the denominator polynomial. Since
            # here the bias term serves as the multiplier for each channel for qN, we can discard it assuming that if any 
            # amplification is necessary, the gradient descent should modify the weights accordingly.
            if degrees[0] == 0:
                view_dims = (1, 1, -1) if qN.ndim == 3 else (1, -1, 1, 1)
                return qN*pm_bias.view(*view_dims), qN**2 + qn**2 # qN*pm_bias, ...
            
            if degrees[0] == 1:
                view_dims = (1, 1, -1) if pm_tuple.ndim == 3 else (1, -1, 1, 1)
                pM = pm_tuple + pm_bias.view(*view_dims)
                return qN*pM + qn*pm_bias.view(*view_dims), qN**2 + qn**2 # ... + qn*pm_bias, ...
            
            view_dims = (1, 1, -1) if pm_tuple[0].ndim == 3 else (1, -1, 1, 1)
            pM = pm_tuple[0] + pm_bias.view(*view_dims)
            pm = pm_tuple[1] + pm_bias.view(*view_dims)
            return qN*pM + qn*pm, qN**2 + qn**2

    # ______________________________________________________________________________________________________________________
    #       
    def forward(self, x, offset=None):
        x_m = self.prepare_polynomial(x, 0, self.get_m_poly, offset) if self.degrees[0] != 0 else None
        x_n = self.prepare_polynomial(x, 1, self.get_n_poly, offset) if self.degrees[1] != 0 else None

        # If we don't have a denominator (N = 0), then directly return the Taylor series expansion, which is the first 
        # entity in the x_m tuple. We also know that both M and N cannot be 0 at the same time
        if x_n is None: 
            return x_m + self.w0
        
        # Prepare the polynomials for Paon-(A/S)
        x_m, x_n = self.prepare_for_div(x_m, self.w0, x_n, self.degrees)
        return torch.div(x_m, x_n)
    
    # ______________________________________________________________________________________________________________________
    #
    @staticmethod
    def __repr__(s):
        def addindent(s_, numSpaces):
            s = s_.split('\n')
            first = s.pop(0)
            s = [(numSpaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = s[:-1] + '\n)'
            s = first + '\n' + s
            return s
        return addindent(s, 2)
# endregion

# ==========================================================================================================================
#region Fully Connected with PaLa
# ==========================================================================================================================

class PaLaLinear(PaLaBase):
    r"""
    in_ch: [integer]
        Number of input channels. 
    out_ch: [integer]
        Number of output channels. 
    degrees: Tuple[integer]
        Truncation degrees of the Padé layer for numerator and denominator, respectively.
    paon_type: [string]
        The Paon type to be used in the layer. "a" for Paon-A (absolute), "s" for Paon-S (smooth).
    """
    def __init__(self, in_ch: int, out_ch: int, degrees: tuple[int, int], paon_type: str):

        # Initialize with the base class
        super().__init__(in_ch, out_ch, degrees, paon_type)

        # Get polynomial generators
        if degrees[0] > 0: self.get_m_poly = nn.Parameter(torch.Tensor(degrees[0], in_ch, out_ch))
        if degrees[1] > 0: self.get_n_poly = nn.Parameter(torch.Tensor(degrees[1], in_ch, out_ch))

        self.reset_parameters()

    # ______________________________________________________________________________________________________________________
    # 
    def reset_parameters(self):
        # w0 term initialization
        fan_in = self.in_ch
        bound  = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.w0, -bound, bound)

        # Weight initialization
        for k, degree in enumerate(self.degrees):
            # If the degree is 0, then there is nothing to initialize
            if degree == 0:
                continue

            if k == 0:
                nn.init.kaiming_uniform_(self.get_m_poly[0], a=0, mode='fan_in', nonlinearity='linear')
                if degree > 1:
                    nn.init.zeros_(self.get_m_poly[1:])
            else:
                nn.init.zeros_(self.get_n_poly)

    # ______________________________________________________________________________________________________________________
    # 
    def prepare_polynomial(self, x, n, get_poly, offset=None):
        # Prepare the input
        x = torch.cat([x[:, None]**k for k in range(1, self.degrees[n]+1)], dim=1)

        # Perform matrix multiplication
        x = torch.einsum('bgi,gio->bgo', x, get_poly)
        return self.after_poly(x, n)
    
    # ______________________________________________________________________________________________________________________
    #
    def forward(self, x):
        return super().forward(x)
    
    # ______________________________________________________________________________________________________________________
    # 
    def __repr__(self):
        new_line = "\n"
        info_str = (
            f"{self.__class__.__name__}({new_line}in_ch={self.in_ch}, out_ch={self.out_ch}, "
            f"degrees={self.degrees}, paon_type={self.paon_type}"
        )
        return super().__repr__(info_str)
#endregion

# ==========================================================================================================================
#region Custom Convolution with PaLa
# ==========================================================================================================================

class PaLaConv2d(PaLaBase):
    r"""
    in_ch: [integer]
        Number of input channels. 
    out_ch: [integer]
        Number of output channels. 
    kernel_size: [integer]
        Kernel height and width.  
    degrees: Tuple[integer]
        Truncation degrees of the Padé layer for numerator and denominator, respectively.
    paon_type: [string]
        The Paon type to be used in the layer. "a" for Paon-A (absolute), "s" for Paon-S (smooth).
    bias_range: [integer]
        The range in which the optimal kernel shift is evaluated; br.
            br > 0 ==> The limitation will be clip(shift, -br, br). 
            br = 0 ==> There is no bound on the shifts. 
            br < 0 ==> There is no shift. 
    shift_is_tensor: [bool]
        If False, the shifts are definitely learned through a small network; otherwise it depends on the bias_range being
        equal to 0 (False) or bigger than 0 (True).
    bias_learn: [bool]
        True for learnable, False for constant shifts.
    bias_round: [bool]
        If True, then the center shifts will be rounded to the nearest integer. 
    conv_padding_mode: [string]
        Padding mode for the convolution. It must be one of "zeros", "reflect", "replicate", "circular".
    shift_padding_mode: [string]
        Padding mode for the shifter. It must be one of "zeros", "border", "reflection".
    stride: [integer] 
        The center jumps during the convolutions. Default: 1.
    pad: [integer]
        The number of pixels that are added to the borders. If it is -1, then it will be calculated as kernel_size//2. 
        Default: -1.
    separable: [bool]
        True for depth-wise, False for custom convolution. Default: False.
    wnorm: [bool]
        If True, the main kernels are subject to weight normalization (https://arxiv.org/abs/1602.07868). Default: False.
    """
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, degrees: tuple[int, int], paon_type: str,
        bias_range: int, shift_is_tensor: bool, bias_learn: bool, bias_round: bool, conv_padding_mode: str, 
        shift_padding_mode: str, stride: int = 1, pad: int = -1, separable: bool = False, wnorm: bool = False
    ):
        # Initialize with the base class
        super().__init__(in_ch, out_ch, degrees, paon_type)

        # Checks
        assert conv_padding_mode in {"zeros", "reflect", "replicate", "circular"}, (
            f"conv_padding_mode must be one of these: 'zeros', 'reflect', 'replicate', 'circular'; "
            f"but got {conv_padding_mode}."
        )
        assert shift_padding_mode in {"zeros", "border", "reflection"}, (
            f"shifter_padding_mode must be one of these: 'zeros', 'border', 'reflection'; "
            f"but got {shift_padding_mode}."
        )

        # Attributes
        kernel_size_1x1         = kernel_size == 1
        self.kernel_size_1x1    = kernel_size_1x1

        self.kernel_size        = kernel_size
        self.bias_range         = bias_range
        self.shift_is_tensor    = shift_is_tensor
        self.bias_learn         = bias_learn
        self.bias_round         = bias_round
        self.conv_padding_mode  = conv_padding_mode
        self.shift_padding_mode = shift_padding_mode
        self.stride             = stride
        self.padding            = kernel_size//2 if pad == -1 else pad
        self.separable          = separable if not kernel_size_1x1 else False
        self.wnorm              = wnorm

        # Shifter
        self.shifter = self.get_shifter()
        
        # Get polynomial generators
        if kernel_size_1x1 or not separable:
            if degrees[0] > 0: self.get_m_poly = self.get_custom_conv(0)
            if degrees[1] > 0: self.get_n_poly = self.get_custom_conv(1)
        # Depth-wise separable convolution
        elif separable:
            if degrees[0] > 0: self.get_m_poly = self.get_depth_separable_conv(0)
            if degrees[1] > 0: self.get_n_poly = self.get_depth_separable_conv(1)
        else:
            raise NotImplementedError
        
        # Reset parameters
        self.reset_parameters()

    # ______________________________________________________________________________________________________________________
    # 
    def get_shifter(self):
        return (
            Shifter(
                self.in_ch, self.bias_range, self.shift_is_tensor, self.bias_learn, self.bias_round, self.shift_padding_mode
            )
            if self.bias_range >= 0 else nn.Identity()
        )
    
    # ______________________________________________________________________________________________________________________
    # 
    def get_custom_conv(self, n):
        # First, take the degree
        degree = self.degrees[n]

        # Determine the number of channels and groups
        in_ch = degree*self.in_ch
        if self.paon_type == "a" and n == 0:
            out_ch, groups = self.out_ch, 1
        else:
            out_ch, groups = degree*self.out_ch, degree

        # Convolution
        conv = nn.Conv2d(
            in_ch, out_ch, self.kernel_size, stride=self.stride, bias=False, groups=groups, 
            padding=self.padding, padding_mode=self.conv_padding_mode
        )
        return weight_norm(conv) if self.wnorm else conv

    # ______________________________________________________________________________________________________________________
    # 
    def get_depth_separable_conv(self, n):
        # First, take the degree
        degree = self.degrees[n]

        # Determine the number of channels and groups
        in_ch = degree*self.in_ch
        if self.paon_type == "a" and n == 0:
            out_ch, p_groups, d_groups = self.out_ch, 1, self.out_ch
        else:
            out_ch, p_groups, d_groups = degree*self.out_ch, degree, degree*self.out_ch

        # Convolution
        pconv = nn.Conv2d(in_ch, out_ch, 1, groups=p_groups, bias=False)
        dconv = nn.Conv2d(
            out_ch, out_ch, self.kernel_size, stride=self.stride, bias=False, groups=d_groups, 
            padding=self.padding, padding_mode=self.conv_padding_mode
        )
        if self.wnorm:
            dconv = weight_norm(dconv)
        return nn.Sequential(pconv, dconv)
    
    # ______________________________________________________________________________________________________________________
    # 
    def reset_parameters(self):
        # w0 term initialization
        fan_in = self.in_ch * self.kernel_size**2 # written from the original source code
        bound  = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.w0, -bound, bound)

        # Depth-wise separable convolution
        if self.separable:
            for k, degree in enumerate(self.degrees):
                # If the degree is 0, then there is nothing to initialize
                if degree == 0:
                    continue

                if k == 0:
                    for layer in self.get_m_poly:
                        nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                        if degree > 1:
                            nn.init.zeros_(layer.weight[self.out_ch:])
                else:
                    for layer in self.get_n_poly:
                        # nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                        nn.init.zeros_(layer.weight)
        # Custom convolution
        else:
            for k, degree in enumerate(self.degrees):
                # If the degree is 0, then there is nothing to initialize
                if degree == 0:
                    continue

                if k == 0:
                    nn.init.kaiming_uniform_(self.get_m_poly.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                    if degree > 1:
                        nn.init.zeros_(self.get_m_poly.weight[self.out_ch:])
                else:
                    # nn.init.kaiming_uniform_(self.n_conv.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                    nn.init.zeros_(self.get_n_poly.weight)

        # Shifter
        if not isinstance(self.shifter, nn.Identity): 
            self.shifter.reset_parameters()

    # ______________________________________________________________________________________________________________________
    # 
    def prepare_polynomial(self, x, n, get_poly, offset=None):
        # Prepare the input
        x = torch.cat([x**k for k in range(1, self.degrees[n]+1)], dim=1)

        # Perform convolution
        x = get_poly(x) 
        return self.after_poly(x, n)
    
    # ______________________________________________________________________________________________________________________
    # 
    def forward(self, x):
        # Shift the input, if necessary
        x = self.shifter(x)
        return super().forward(x)
    
    # ______________________________________________________________________________________________________________________
    # 
    def __repr__(self):
        # Bias options string
        learnable_bias_str = f"bias_learn={self.bias_learn}, "
        bias_opts_str = (
            f"bias_range={self.bias_range if self.bias_range > 0 else 'Unbounded'}, shift_is_tensor={self.shift_is_tensor},"
            f" {learnable_bias_str}bias_round={self.bias_round}, "
        ) if self.bias_range >= 0 else "bias_range=None, "
        
        # Padding string
        padding_str = f"conv_padding_mode={self.conv_padding_mode}, "
        if self.bias_range >= 0:
            padding_str = f"{padding_str}shift_padding_mode={self.shift_padding_mode}, "
            
        # Combine everything
        new_line = "\n"
        info_str = (
            f"{self.__class__.__name__}({new_line}in_ch={self.in_ch}, out_ch={self.out_ch}, "
            f"kernel_size={self.kernel_size}, degrees={self.degrees}, paon_type={self.paon_type},{new_line}"
            f"{bias_opts_str}{padding_str}{new_line}"
            f"stride={self.stride}, padding={self.padding}, separable={self.separable})"
        )
        return super().__repr__(info_str)
#endregion
    
# ==========================================================================================================================
#region Transposed Convolution with PaLa
# ==========================================================================================================================

class PaLaConvTranspose2d(PaLaConv2d):
    r"""
    in_ch: [integer]
        Number of input channels. 
    out_ch: [integer]
        Number of output channels. 
    kernel_size: [integer]
        Kernel height and width.  
    degrees: Tuple[integer]
        Truncation degrees of the Padé layer for numerator and denominator, respectively.
    paon_type: [string]
        The Paon type to be used in the layer. "a" for Paon-A (absolute), "s" for Paon-S (smooth).
    bias_range: [integer]
        The range in which the optimal kernel shift is evaluated; br.
            br > 0 ==> The limitation will be clip(shift, -br, br). 
            br = 0 ==> There is no bound on the shifts. 
            br < 0 ==> There is no shift. 
    shift_is_tensor: [bool]
        If False, the shifts are definitely learned through a small network; otherwise it depends on the bias_range being
        equal to 0 (False) or bigger than 0 (True).
    bias_learn: [bool]
        True for learnable, False for constant shifts. Valid only when deformable = False.
    bias_round: [bool]
        If True, then the center shifts will be rounded to the nearest integer. 
    conv_padding_mode: [string]
        Padding mode for the convolution. It must be "zeros".
    shift_padding_mode: [string]
        Padding mode for the shifter. It must be one of "zeros", "border", "reflection".
    output_padding: [integer]
        Output padding for transposed convolution. 
    stride: [integer] 
        The center jumps during the convolutions. Default: 1.
    pad: [integer]
        The number of pixels that are added to the borders. If it is -1, then it will be calculated as kernel_size//2. 
        Default: -1.
    separable: [bool]
        True for depth-wise, False for custom convolution. Default: False.
    wnorm: [bool]
        If True, the main kernels are subject to weight normalization (https://arxiv.org/abs/1602.07868). Default: False.
    """
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, degrees: tuple[int, int], paon_type: str,
        bias_range: int, shift_is_tensor: bool, bias_learn: bool, bias_round: bool, 
        conv_padding_mode: str, shift_padding_mode: str, output_padding: int, 
        stride: int = 1, pad: int = -1, separable: bool = False, wnorm: bool = False
    ):
        # Check
        assert conv_padding_mode == "zeros", \
            f"conv_padding_mode must be 'zeros' for transposed; but got {conv_padding_mode}."
        
        # Attributes
        self.output_padding = output_padding
        
        # Initialize with PaLaConv2d
        super().__init__(
            in_ch, out_ch, kernel_size, degrees, paon_type, bias_range, shift_is_tensor, bias_learn, bias_round, 
            conv_padding_mode, shift_padding_mode, stride, pad, separable, wnorm
        )
        
    # ______________________________________________________________________________________________________________________
    # 
    def get_custom_conv(self, n):
        # First, take the degree
        degree = self.degrees[n]

        # Determine the number of channels and groups
        in_ch = degree*self.in_ch
        if self.paon_type == "a" and n == 0:
            out_ch, groups = self.out_ch, 1
        else:
            out_ch, groups = degree*self.out_ch, degree
            
        conv = nn.ConvTranspose2d(
            in_ch, out_ch, self.kernel_size, stride=self.stride, bias=False, groups=groups, 
            padding_mode=self.conv_padding_mode, padding=self.padding, output_padding=self.output_padding
        )
        return weight_norm(conv) if self.wnorm else conv

    # ______________________________________________________________________________________________________________________
    # 
    def get_depth_separable_conv(self, n):
        # First, take the degree
        degree = self.degrees[n]

        # Determine the number of channels and groups
        in_ch = degree*self.in_ch
        if self.paon_type == "a" and n == 0:
            out_ch, p_groups, d_groups = self.out_ch, 1, self.out_ch
        else:
            out_ch, p_groups, d_groups = degree*self.out_ch, degree, degree*self.out_ch

        pconv = nn.Conv2d(in_ch, out_ch, 1, groups=p_groups, bias=False)
        dconv = nn.ConvTranspose2d(
            in_ch, out_ch, self.kernel_size, stride=self.stride, bias=False, groups=d_groups, 
            padding_mode=self.conv_padding_mode, padding=self.padding, output_padding=self.output_padding
        )
        if self.wnorm:
            dconv = weight_norm(dconv)
        return nn.Sequential(pconv, dconv)
    
    # ______________________________________________________________________________________________________________________
    # 
    def __repr__(self):
        # Bias options string
        learnable_bias_str = f"bias_learn={self.bias_learn}, "
        bias_opts_str = (
            f"bias_range={self.bias_range if self.bias_range > 0 else 'Unbounded'}, shift_is_tensor={self.shift_is_tensor},"
            f" {learnable_bias_str}bias_round={self.bias_round}, "
        ) if self.bias_range >= 0 else "bias_range=None, "
        
        # Padding string
        padding_str = f"conv_padding_mode={self.conv_padding_mode}, "
        if self.bias_range >= 0:
            padding_str = f"{padding_str}shift_padding_mode={self.shift_padding_mode}, "
        padding_str = f"{padding_str}output_padding={self.output_padding}, "
            
        # Combine everything
        new_line = "\n"
        info_str = (
            f"{self.__class__.__name__}({new_line}in_ch={self.in_ch}, out_ch={self.out_ch}, "
            f"kernel_size={self.kernel_size}, degrees={self.degrees}, paon_type={self.paon_type},{new_line}"
            f"{bias_opts_str}{padding_str}{new_line}"
            f"stride={self.stride}, padding={self.padding}, separable={self.separable})"
        )
        return super(PaLaConv2d, self).__repr__(info_str)
#endregion

# ==========================================================================================================================
#region Deformation Offset
# ==========================================================================================================================

class DeformOffset(nn.Module):
    r"""
    in_ch: [integer]
        The number of channels of the features to be shifted.
    kernel_size: [integer]
        The size of the kernel that the shifts are calculated for.
    bias_range: [integer]
        The range in which the optimal kernel shift is evaluated.
    bias_round: [bool]
        If True, then the center shifts will be rounded to the nearest integer.
    offset_kernel: [integer]
        Kernel height and width for offset calculating convolution.
    stride: [integer]
        The stride value for the offset-calculating convolution.
    shift_padding_mode: [string]
        Padding mode for the shifter. It must be one of "zeros", "border", "reflection".
    full_deform: [bool]
        If True, then deformation will be calculated for each kernel element, as in the regular deformable convolution.
    channelwise: [bool]
        If True, then offsets are calculated for each channel separately, increasing the computation time and number of 
        parameters.
    """
    def __init__(
        self, in_ch: int, kernel_size: int, bias_range: int, bias_round: bool, offset_kernel: int, stride: int, 
        shift_padding_mode: str, full_deform: bool, channelwise: bool
    ):
        super().__init__()

        # Attributes
        self.kernel_size = kernel_size
        self.full_deform = full_deform
        self.bias_range = bias_range
        self.bias_round = bias_round

        # Offset convolution settings
        out_ch = 2
        if channelwise: out_ch *= in_ch
        if full_deform: out_ch *= kernel_size**2
        
        self.offset_conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=offset_kernel, stride=stride, 
            padding=offset_kernel//2, padding_mode=shift_padding_mode, dilation=1, bias=True
        )

        self.reset_parameters()

    # ----------------------------------------------------------------------------------------------------------------------
    def reset_parameters(self):
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    # ----------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        # Take the input device
        dvc = x.device

        # Find the offset values
        offset_r = self.offset_conv(x)
        offset = torch.zeros_like(offset_r)

        # Take the offset shape
        h, w = offset.shape[2:]
        
        # Limit the offsets so that they cannot go out of the feature vector
        grid_x, grid_y = torch.meshgrid(torch.arange(w, device=dvc), torch.arange(h, device=dvc), indexing='xy')
        offset[:, 0::2] = torch.clamp(offset_r[:, 0::2], -grid_y, h - grid_y)
        offset[:, 1::2] = torch.clamp(offset_r[:, 1::2], -grid_x, w - grid_x)

        # Limit the offsets so that the maximum shift has a limit inside of the feature vector
        max_shift = self.bias_range if self.bias_range > 0 else max(h, w)/4.
        offset = torch.where(
            torch.abs(offset) >= max_shift, 
            max_shift*torch.tanh(offset / max_shift), 
            offset
        )

        # Round the offsets to the integer values
        if self.bias_round: offset = torch.round(offset)

        # If not full_deform, then we have to match the size of offset to the necessary size
        if not self.full_deform: 
            offset = torch.repeat_interleave(offset, self.kernel_size**2, 1)
        return offset
#endregion
    
# ==========================================================================================================================
#region Deformable Convolution with PaLa
# ==========================================================================================================================

class PaLaDeformConv2d(PaLaBase):
    r"""
    in_ch: [integer]
        Number of input channels. 
    out_ch: [integer]
        Number of output channels. 
    kernel_size: [integer]
        Kernel height and width.  
    degrees: Tuple[integer]
        Truncation degrees of the Padé layer for numerator and denominator, respectively.
    paon_type: [string]
        The Paon type to be used in the layer. "a" for Paon-A (absolute), "s" for Paon-S (smooth).
    bias_range: [integer]
        The range in which the optimal kernel shift is evaluated; br.
            br > 0  ==> The limitation will be br*tanh(shift/br).
            br <= 0 ==> Then br = max(h, w)/4, where h and w are calculated from input.
    bias_round: [bool]
        If True, then the center shifts will be rounded to the nearest integer. 
    conv_padding_mode: [string]
        Padding mode for the convolution. It must be one of "constant", "reflect", "replicate", "circular".
    shift_padding_mode: [string]
        Padding mode for the offset calculation. It must be one of "zeros", "reflect", "replicate", "circular".
    full_deform: [bool]
        If True, then deformation will be calculated for each kernel element, as in the regular deformable convolution. 
        Default: True.
    channelwise: [bool]
        If True, then offsets are calculated for each channel separately, increasing the computation time and number of 
        parameters. Default: False.
    offset_kernel: [integer]
        Kernel height and width for offset calculating convolution. Default: 1.
    stride: [integer] 
        The center jumps during the convolutions. Default: 1.
    pad: [integer]
        The number of pixels that are added to the borders. If it is -1, then it will be calculated as kernel_size//2. 
        Default: -1.
    separable: [bool]
        True for depth-wise, False for custom convolution. Default: False.
    wnorm: [bool]
        If True, the main kernels are subject to weight normalization (https://arxiv.org/abs/1602.07868). Default: False.
    """
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, degrees: tuple[int, int], paon_type: str,
        bias_range: int, bias_round: bool, conv_padding_mode: str, shift_padding_mode: str,
        full_deform: bool = True, channelwise: bool = False, offset_kernel: int = 1,
        stride: int = 1, pad: int = -1, separable: bool = False, wnorm: bool = False
    ):
        # Initialize with the base class
        super().__init__(in_ch, out_ch, degrees, paon_type)

        # Checks
        assert kernel_size != 1, "For deformable convolution, kernel_size=1 is meaningless."
        assert conv_padding_mode in {"constant", "reflect", "replicate", "circular"}, (
            f"conv_padding_mode must be one of these: 'constant', 'reflect', 'replicate', 'circular'; "
            f"but got {conv_padding_mode}."
        )
        assert shift_padding_mode in {"zeros", "reflect", "replicate", "circular"}, (
            f"shifter_padding_mode must be one of these: 'zeros', 'reflect', 'replicate', 'circular'; "
            f"but got {shift_padding_mode}."
        )

        # Attributes
        kernel_size_1x1         = kernel_size == 1
        self.kernel_size_1x1    = kernel_size_1x1

        self.kernel_size        = kernel_size
        self.bias_range         = bias_range
        self.bias_round         = bias_round
        self.conv_padding_mode  = conv_padding_mode
        self.shift_padding_mode = shift_padding_mode
        self.full_deform        = full_deform 
        self.channelwise        = channelwise
        self.offset_kernel      = offset_kernel
        self.stride             = stride
        self.padding            = kernel_size//2 if pad == -1 else pad
        self.separable          = separable if not kernel_size_1x1 else False
        self.wnorm              = wnorm

        # Shifter
        self.shifter = self.get_shifter()

        # Get polynomial generators
        if kernel_size_1x1 or not separable:
            if degrees[0] > 0: self.get_m_poly = self.get_custom_conv(0)
            if degrees[1] > 0: self.get_n_poly = self.get_custom_conv(1)
        # Depth-wise separable convolution
        elif separable:
            if degrees[0] > 0: self.get_m_poly = self.get_depth_separable_conv(0)
            if degrees[1] > 0: self.get_n_poly = self.get_depth_separable_conv(1)
        else:
            raise NotImplementedError

        # Polynomial preparation changes for separable convolution
        if separable:
            self.prepare_polynomial = self.prepare_polynomial_separable
        else:
            self.prepare_polynomial = self.prepare_polynomial_custom

        # after_conv operation
        if paon_type == "a":
            self.after_poly = super().after_poly_a
        elif paon_type == "s":
            self.after_poly = super().after_poly_s
        else:
            raise NotImplementedError
        
        self.reset_parameters()

    # ______________________________________________________________________________________________________________________
    # 
    def get_shifter(self):
        return DeformOffset(
            self.in_ch, self.kernel_size, self.bias_range, self.bias_round, self.offset_kernel, self.stride, 
            self.shift_padding_mode, self.full_deform, self.channelwise
        )
    
    # ______________________________________________________________________________________________________________________
    # 
    def get_custom_conv(self, n):
        # First, take the degree
        degree = self.degrees[n]

        # Determine the number of channels and groups
        in_ch = degree*self.in_ch
        if self.paon_type == "a" and n == 0:
            out_ch, groups = self.out_ch, 1
        else:
            out_ch, groups = degree*self.out_ch, degree

        # Convolution
        conv = DeformConv2d(
            in_ch, out_ch, self.kernel_size, stride=self.stride, bias=False, groups=groups, padding=0
        )
        return weight_norm(conv) if self.wnorm else conv

    # ______________________________________________________________________________________________________________________
    # 
    def get_depth_separable_conv(self, n):
        # First, take the degree
        degree = self.degrees[n]

        # Determine the number of channels and groups
        in_ch = degree*self.in_ch
        if self.paon_type == "a" and n == 0:
            out_ch, p_groups, d_groups = self.out_ch, 1, self.out_ch
        else:
            out_ch, p_groups, d_groups = degree*self.out_ch, degree, degree*self.out_ch

        # Convolution
        pconv = nn.Conv2d(in_ch, out_ch, 1, groups=p_groups, bias=False)
        dconv = DeformConv2d(
            out_ch, out_ch, self.kernel_size, stride=self.stride, bias=False, groups=d_groups, padding=0
        )
        if self.wnorm:
            dconv = weight_norm(dconv)
        return nn.Sequential(pconv, dconv)
    
    # ______________________________________________________________________________________________________________________
    # 
    def reset_parameters(self):
        # w0 term initialization
        fan_in = self.in_ch * self.kernel_size**2 # written from the original source code
        bound  = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.w0, -bound, bound)

        # Depth-wise separable convolution
        if self.separable:
            for k, degree in enumerate(self.degrees):
                # If the degree is 0, then there is nothing to initialize
                if degree == 0:
                    continue

                if k == 0:
                    for layer in self.get_m_poly:
                        nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                        if degree > 1:
                            nn.init.zeros_(layer.weight[self.out_ch:])
                else:
                    for layer in self.get_n_poly:
                        # nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                        nn.init.zeros_(layer.weight)
        # Custom convolution
        else:
            for k, degree in enumerate(self.degrees):
                # If the degree is 0, then there is nothing to initialize
                if degree == 0:
                    continue

                if k == 0:
                    nn.init.kaiming_uniform_(self.get_m_poly.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                    if degree > 1:
                        nn.init.zeros_(self.get_m_poly.weight[self.out_ch:])
                else:
                    # nn.init.kaiming_uniform_(self.n_conv.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                    nn.init.zeros_(self.get_n_poly.weight)

        # Shifter
        if not isinstance(self.shifter, nn.Identity): 
            self.shifter.reset_parameters()

    # ______________________________________________________________________________________________________________________
    # 
    def prepare_polynomial_custom(self, x, n, get_poly, offset=None):
        # Prepare and pad the input since deformable has no padding, as it is set such
        x = torch.cat([x**k for k in range(1, self.degrees[n]+1)], dim=1)
        p = tuple([self.kernel_size//2 for _ in range(4)])
        x = F.pad(x, mode=self.conv_padding_mode, pad=p)

        # Perform deformable convolution
        x = get_poly(x, offset, None)
        return self.after_poly(x, n)
    
    # ______________________________________________________________________________________________________________________
    # 
    def prepare_polynomial_separable(self, x, n, get_poly, offset=None):
        # Prepare and pad the input since deformable has no padding, as it is set such
        x = torch.cat([x**k for k in range(1, self.degrees[n]+1)], dim=1)
        p = tuple([self.kernel_size//2 for _ in range(4)])
        x = F.pad(x, mode=self.conv_padding_mode, pad=p)

        # Perform deformable convolution
        x = get_poly[1](get_poly[0](x), offset, None)
        return self.after_poly(x, n)
    
    # ______________________________________________________________________________________________________________________
    # 
    def forward(self, x):
        offset = self.shifter(x)
        return super().forward(x, offset)
    
    # ______________________________________________________________________________________________________________________
    # 
    def __repr__(self):
        # Bias options string
        bias_opts_str = (
            f"bias_range={self.bias_range if self.bias_range > 0 else 'max(h,w)/4'}, bias_round={self.bias_round}, "
        )
        
        # Padding string
        padding_str = f"conv_padding_mode={self.conv_padding_mode}, shift_padding_mode={self.shift_padding_mode}, "
        
        # Deformable string
        deformable_str = (
            f"full_deform={self.full_deform}, channelwise={self.channelwise}, offset_kernel={self.offset_kernel}, "
        )
            
        # Combine everything
        new_line = "\n"
        info_str = (
            f"{self.__class__.__name__}({new_line}in_ch={self.in_ch}, out_ch={self.out_ch}, "
            f"kernel_size={self.kernel_size}, degrees={self.degrees}, paon_type={self.paon_type},{new_line}"
            f"{bias_opts_str}{padding_str}{new_line + deformable_str + new_line}"
            f"stride={self.stride}, padding={self.padding}, separable={self.separable})"
        )
        return super().__repr__(info_str)
#endregion