# %%
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
# %%
seed = 0
gridw = 4
gridh = 4
device = torch.device('cuda')
network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl'

sigma_min = 0.002
sigma_max = 80
num_steps = 18
rho = 7

S_churn = 0
S_min = 0
S_max = float('inf')
S_noise = 1
# %%
batch_size = gridw * gridh
torch.manual_seed(seed)

print(f'Loading network from "{network_pkl}"...')
with dnnlib.util.open_url(network_pkl) as f:
    net = pickle.load(f)['ema'].to(device)
# %%
print(f'Generating {batch_size} images...')
latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
class_labels = None
if net.label_dim:
    class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

# Adjust noise levels based on what's supported by the network.
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)

# Time step discretization.
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
# %%
# Main sampling loop.
x_next = latents.to(torch.float64) * t_steps[0]
for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
    x_cur = x_next
    print(i, torch.mean(x_cur).item(), torch.std(x_cur).item(), t_cur.item())

    # Increase noise temporarily.
    # S_churn is 0, so gamma is 0
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    # t_hat = t_cur
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    # x_hat = x_hat + S_noise * noise
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

    # Euler step.
    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next, class_labels).to(torch.float64)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
image = image.cpu().numpy()
PIL.Image.fromarray(image, 'RGB')
# %%
x_0 = x_next
# %%
# rand_t = torch.randint(0, num_steps, ())
rand_t = 5
print(f"Starting at sigma = {t_steps[rand_t]:.3f}")
new_t_steps = t_steps[t_steps <= t_steps[rand_t]]

n = torch.randn_like(x_0) * t_steps[rand_t]
x_next = x_0 + n
image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
image = image.cpu().numpy()
PIL.Image.fromarray(image, 'RGB')
# %%
for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(new_t_steps[:-1], new_t_steps[1:]))), unit='step'): # 0, ..., N-1
    x_cur = x_next

    # Increase noise temporarily.
    # S_churn is 0, so gamma is 0
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    # t_hat = t_cur
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    # x_hat = x_hat + S_noise * noise
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

    # Euler step.
    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < len(new_t_steps) - 2:
        denoised = net(x_next, t_next, class_labels).to(torch.float64)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
image = image.cpu().numpy()
PIL.Image.fromarray(image, 'RGB')
# %%
