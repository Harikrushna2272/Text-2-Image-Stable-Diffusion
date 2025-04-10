import torch
import numpy as np

class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        # generator: A PyTorch random number generator, used for deterministic noise generation.
        # num_training_steps: The number of training diffusion steps (default: 1000).
        
        # beta_start and beta_end: Define the noise schedule, controlling how noise is added during training.
        # Generates a sequence of num_training_steps values between beta_start and beta_end (both square-rooted and squared again).
        # These values control the noise variance added to images at each step.
        # They determine how much noise is added at each step of the diffusion
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        # Determines the amount of information retained after each diffusion step.
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # The cumulative product of alphas, used to track the total noise accumulated at each step.
        self.one = torch.tensor(1.0)
        # A helper tensor used in calculations.

        self.generator = generator
        # self.generator: Stores the random number generator.

        self.num_train_timesteps = num_training_steps
        # self.num_train_timesteps: Stores the number of training timesteps.

        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        # self.timesteps: Generates an array of timesteps in descending order (from num_training_steps-1(1000) to 0). to reduce noice from high to low.

    def set_inference_timesteps(self, num_inference_steps=50):
    # This function reduces the number of timesteps used for inference (e.g., from 1000 to 50) for faster sampling.
        # like 999, 999-50=920, so on and so far,... 0 = 50 steps.
        self.num_inference_steps = num_inference_steps
         
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # self.num_train_timesteps = 1000 (total training steps).
        # self.num_inference_steps = 50 (inference steps).
        # step_ratio = 1000 // 50 = 20
        # This means that each inference step will be taken after skipping 20 training steps.

        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        # np.arange(0, num_inference_steps): Creates an array [0, 1, 2, ..., 49].
        # Multiply by step_ratio = 20, giving
        # [::-1]: Reverses the order
        # This ensures we start from the most noised step (t=980) and go down to t=0 (final output)

        self.timesteps = torch.from_numpy(timesteps)
        # Converts the NumPy array to a PyTorch tensor for further computations.

    def _get_previous_timestep(self, timestep: int) -> int:
    # This function calculates the variance for noise addition at each timestep.


        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)
        # This calculates the variance for noise addition at each timestep using above 
        # function.

        alpha_prod_t = self.alphas_cumprod[timestep]
        # Cumulative product of alpha values at timestep.
        # Represents how much original signal remains at timestep (without noise).

        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        # Gets the cumulative product of alpha at the previous timestep.
        # If prev_t < 0, it sets it to 1.0 (since at t=0, noise should be 0).
        
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        # This measures the amount of noise added at t.

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        # Uses formula from the DDPM paper (Eq. 7) to compute variance for noise addition.

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)
        # Ensures that variance never becomes zero (which would cause division errors).

        return variance
    
    def set_strength(self, strength=1):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        # self.num_inference_steps is the total number of steps (e.g., 50).
        # int(self.num_inference_steps * strength): Computes the number of steps to keep based on strength.
        # Subtracting this from self.num_inference_steps gives start_step, which is the index from which denoising starts.

        self.timesteps = self.timesteps[start_step:]
        # self.timesteps contains the selected inference timesteps.
        # It removes earlier steps to skip part of the denoising process.

        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
    # This function performs a single step of denoising at timestep t, updating the latent variable.
        t = timestep # Retrieves the current timestep 
        prev_t = self._get_previous_timestep(t) # Computes the previous timestep

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        # alpha_prod_t: Cumulative alpha product at t, representing how much signal remains.
        # alpha_prod_t_prev: Same for the previous timestep (prev_t)
        # beta_prod_t: Amount of noise at timestep t.
        # beta_prod_t_prev: Amount of noise at previous timestep.
        # current_alpha_t: Transition factor between t and t-1.
        # current_beta_t: Noise change factor.

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        # represents the reconstructed original image before noise was added.

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # the variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
    # This function adds controlled noise to the input original_samples (clean images or latents) based on a specific timestep t
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from q(x_t | x_0) as in equation (4) of https://arxiv.org/pdf/2006.11239.pdf
        # Because N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # here mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    # Adds noise to a given clean image x_0 at a given timestep t.
    # Uses precomputed alphas_cumprod to determine how much signal vs. noise remains at t.
    # Uses random Gaussian noise to create a controlled noisy sample.
    # Ensures proper shape alignment for tensor operations.

        

    