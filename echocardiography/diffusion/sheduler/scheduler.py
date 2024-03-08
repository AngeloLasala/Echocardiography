
"""
Import the linear noise scheduler from the DDPM paper

https://github.com/explainingai-code/DDPM-Pytorch/blob/main/scheduler/linear_noise_scheduler.py
"""
import torch


class LinearNoiseScheduler:
    """
    Class for the linear noise scheduler that is used in DDPM.
    """
    def __init__(self, num_timesteps, beta_start, beta_end, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(self.device)
        alphas = 1. - self.betas
        self.alphas = alphas.to(self.device)
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod).to(self.device)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod).to(self.device)
        
    def add_noise(self, original, noise, t):
        """
        Forward method for diffusion: xt = sqrt(cum_alpha_t) * x0 + sqrt(1 - cum_alpha_t) * noise

        Parameters
        ----------
        original: Image on which noise is to be applied
        noise: Random Noise Tensor (from normal dist)
        t: timestep of the forward process of shape -> (B,)
        
        Return
        ------
        Image with noise applied
        """
        original_shape = original.shape
        t = t.to(original.device)
        noise = noise.to(original.device)
        batch_size = original_shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size).to(original.device)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size).to(original.device)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape)-1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
        
    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        Use the noise prediction by model to get
        xt-1 using xt and the nosie predicted: xt-1 = mu_t + sigma_t * z

        Parameters
        ----------
        xt: current timestep sample
        noise_pred: model noise prediction
        t: current timestep we are at
        :return:
        """
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / torch.sqrt(self.alpha_cum_prod[t])
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas[t])*noise_pred)/(self.sqrt_one_minus_alpha_cum_prod[t])
        mean = mean / torch.sqrt(self.alphas[t])
        
        if t == 0:
            return mean, mean
        else:
            variance = (1-self.alpha_cum_prod[t-1]) / (1.0 - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma*z, x0

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = torch.zeros(8, 1, 56, 56)
    ## create a mask for a square of 1
    mask = torch.zeros(8, 1, 56, 56)
    mask[:, :, 20:30, 20:30] = 1
    
    plt.figure()
    plt.imshow(mask[0, 0].cpu().numpy())

    ## add noise to the image
    sheduler = LinearNoiseScheduler(1000, 0.0001, 0.001)
    image = mask.to(device) 
    noise = torch.randn_like(image)
    t = torch.tensor([0 for _ in range(8)])
    add_noise = sheduler.add_noise(image, noise, t)
    print(add_noise.shape)

    for i in [0,10,100,999]:
        t = torch.tensor([i for _ in range(8)])
        add_noise = sheduler.add_noise(image, noise, t)
        plt.figure(num=f'add noise time {i}')
        plt.imshow(add_noise[0, 0].cpu().numpy())

    plt.show()