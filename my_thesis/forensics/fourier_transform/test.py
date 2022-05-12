import torch

x = torch.tensor([0, 0.25, -0.5, -0.25], dtype=torch.float32).view(2, 2)
shifted = torch.fft.fftshift(x)
print(x, "\n", shifted)

y = torch.fft.fftfreq(4).view(2, 2)
print(y, "\n", torch.fft.fftshift(y))
print()

x_ = x.unsqueeze_(0)
z = torch.cat([x, x, x], dim=0)
print(z)
z_shifted = torch.fft.fftshift(z)
print(z_shifted)
ishifted = torch.fft.ifftshift(z_shifted)
print(ishifted)