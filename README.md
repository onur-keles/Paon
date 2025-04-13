# <div align="center">Paon: <ins>Pa</ins>dé Approximant Neur<ins>on</ins>
The PyTorch implementation for Pade Approximant Neurons, first introduced in "_PAON: A New Neuron Model using Padé Approximants,_" accepted to the _International Conference on Image Processing_ (ICIP), 2024.

---

For linear layer, the example usage is given below.

```py
import torch
from paon import PaLaLinear

# Attributes
linear_dict = {
    'in_ch': 4096,     # Input features
    'out_ch': 4096,    # Output features
    'degrees': (1,1),  # (M, N): Polynomial degrees [M/N]
    'paon_type': "s",  # "s" for Paon-S, "a" for Paon-A
}

# Layer
pala = PaLaLinear(**linear_dict).float().to("cuda")

# Operation
inp = torch.randn(1, 4096).float().to("cuda")
out = pala(inp)

```

For convolutional layer, the example usage is given below.
```py
import torch
from paon import PaLaConv2d

# Attributes
conv_dict = {
    'in_ch': 3,                       # Input channels
    'out_ch': 3,                      # Output channels
    'kernel_size': 5,                 # Convolution kernel size
    'degrees': (1,1),                 # (M, N): Polynomial degrees [M/N]
    'paon_type': "s",                 # "s" for Paon-S, "a" for Paon-A
    'bias_range': 0,                  # Allowable maximum shift; bias_range<0 => No shift, bias_range>0 => Limited, bias_range=0 => Unlimited
    'shift_is_tensor': False,         # Whether directly optimized via back-propagation or learned through one-layer network
    'bias_learn': True,               # Whether the shifts are learnable or constant
    'bias_round': False,              # Whether round the shift values or not
    'conv_padding_mode': "replicate", # Padding for convolution
    'shift_padding_mode': "border",   # Padding for convolution that learns the shifts
}

# Layer
pala = PaLaConv2d(**conv_dict).float().to("cuda")

# Operation
inp = torch.randn(1,3,256,256).float().to("cuda")
out = pala(inp)
```

For transposed convolutional layer, the example usage is given below.
```py
import torch
from paon import PaLaConvTranspose2d

# Attributes
transpose_dict = {
    'in_ch': 3,                       # Input channels
    'out_ch': 3,                      # Output channels
    'kernel_size': 5,                 # Convolution kernel size
    'degrees': (1,1),                 # (M, N): Polynomial degrees [M/N]
    'paon_type': "s",                 # "s" for Paon-S, "a" for Paon-A
    'bias_range': 0,                  # Allowable maximum shift; bias_range<0 => No shift, bias_range>0 => Limited, bias_range=0 => Unlimited
    'shift_is_tensor': False,         # Whether directly optimized via back-propagation or learned through one-layer network
    'bias_learn': True,               # Whether the shifts are learnable or constant
    'bias_round': False,              # Whether round the shift values or not
    'conv_padding_mode': "zeros",     # Padding for convolution; for transposed convolution, zero-padding might be a must
    'shift_padding_mode': "border",   # Padding for convolution that learns the shifts
    'output_padding': 1,              # Transposed convolution output padding
    'stride': 2,                      # Transposed convolution stride
}

# Layer
pala = PaLaConvTranspose2d(**transpose_dict).float().to("cuda")

# Operation
inp = torch.randn(1,3,256,256).float().to("cuda")
out = pala(inp)
```

For deformable convolutional layer, the example usage is given below.
```py
import torch
from paon import PaLaDeformConv2d

# Attributes
deform_dict = {
    'in_ch': 3,                        # Input channels
    'out_ch': 3,                       # Output channels
    'kernel_size': 5,                  # Convolution kernel size
    'degrees': (1,1),                  # (M, N): Polynomial degrees [M/N]
    'paon_type': "s",                  # "s" for Paon-S, "a" for Paon-A
    'bias_range': -1,                  # Allowable maximum shift; bias_range<=0 => No limited, bias_range>0 => Limited
    'bias_round': False,               # Whether round the shift values or not
    'conv_padding_mode': "replicate",  # Padding for convolution
    'shift_padding_mode': "replicate", # Padding for convolution that learns the shifts
    'full_deform': True,               # Whether shift all the kernel elements independently or shift the kernel as a whole
    'channelwise': False,              # Whether calculate separate shift values for each input channel or not
    'offset_kernel': 1,                # Kernel size for the offset calculating convolution
}

# Layer
pala = PaLaDeformConv2d(**deform_dict).float().to("cuda")

# Operation
inp = torch.randn(1,3,256,256).float().to("cuda")
out = pala(inp)
```

---
# Citation
If this repository is helpful to you in any way, please cite the following work:
```
@INPROCEEDINGS{keles2024paon,
  author={Keleş, Onur and Tekalp, A. Murat},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
  title={Paon: A New Neuron Model Using Pad\'e Approximants}, 
  year={2024},
  pages={207-213},
  doi={10.1109/ICIP51287.2024.10648214}
}
```
