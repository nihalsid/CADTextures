import torch


class GANLoss:

    def __init__(self, loss_type, occupancy_checker=None, occupancy_threshold=0.05):
        self.compute_generator_loss = self.compute_generator_loss_wasserstein
        self.compute_discriminator_loss = None
        self.gradient_penalty_computation = True
        self.occupancy_checker = occupancy_checker
        self.occupancy_threshold = occupancy_threshold
        if loss_type == 'hinge':
            self.compute_discriminator_loss = self.compute_discriminator_loss_hinge
        if loss_type == 'vanilla':
            self.compute_discriminator_loss = self.compute_discriminator_loss_vanilla
            self.compute_generator_loss = self.compute_generator_loss_vanilla
        elif loss_type == 'wgan':
            self.compute_discriminator_loss = self.compute_discriminator_loss_wasserstein
        elif loss_type == 'wgan_gp':
            self.compute_discriminator_loss = self.compute_discriminator_loss_wasserstein_gp

    def compute_generator_loss_wasserstein(self, ref_disc, in_fake):
        return -torch.mean(self.adjust_prediction_based_on_occupancy(in_fake, ref_disc(in_fake)))

    def adjust_prediction_based_on_occupancy(self, d_in, d_output):
        if self.occupancy_checker is None:
            mask = torch.ones_like(d_output)
        else:
            mask = (self.occupancy_checker(d_in) > self.occupancy_threshold).float()
        return d_output * mask * (torch.ones_like(d_output).sum() / mask.sum())

    def compute_discriminator_loss_wasserstein(self, ref_disc, in_real, in_fake):
        d_real = ref_disc(in_real)
        d_fake = ref_disc(in_fake)
        real_loss = -torch.mean(self.adjust_prediction_based_on_occupancy(in_real, d_real))
        fake_loss = torch.mean(self.adjust_prediction_based_on_occupancy(in_fake, d_fake))
        return real_loss, fake_loss, 0.000

    @staticmethod
    def compute_discriminator_loss_hinge(ref_disc, in_real, in_fake):
        d_real = ref_disc(in_real)
        d_fake = ref_disc(in_fake)
        real_loss = torch.mean(torch.nn.functional.relu(1. - d_real))
        fake_loss = torch.mean(torch.nn.functional.relu(1. + d_fake))
        return real_loss, fake_loss, torch.Tensor([0]).requires_grad_(True).cuda(in_real.device.index)

    def compute_discriminator_loss_wasserstein_gp(self, ref_disc, in_real, in_fake):
        real_loss, fake_loss, epsilon_loss = self.compute_discriminator_loss_wasserstein(ref_disc, in_real, in_fake)
        gp = (self.compute_gradient_penalty(ref_disc, in_real, in_fake) + epsilon_loss) if self.gradient_penalty_computation else 0
        return real_loss, fake_loss, gp

    @staticmethod
    def compute_discriminator_loss_vanilla(ref_disc, in_real, in_fake):
        d_real = ref_disc(in_real)
        d_fake = ref_disc(in_fake)
        real_labels = torch.ones_like(d_real)
        fake_labels = torch.zeros_like(d_fake)
        real_loss = torch.nn.functional.binary_cross_entropy_with_logits(d_real, real_labels, reduction='none')
        real_loss = torch.mean(real_loss)
        fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(d_fake, fake_labels, reduction='none')
        fake_loss = torch.mean(fake_loss)
        return real_loss, fake_loss, torch.Tensor([0]).requires_grad_(True).cuda(in_real.device.index)

    @staticmethod
    def compute_generator_loss_vanilla(ref_disc, in_fake):
        d_fake = ref_disc(in_fake)
        fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(d_fake, torch.ones(d_fake.shape).cuda()).mean()
        return fake_loss

    @staticmethod
    def compute_gradient_penalty(ref_disc, in_real, in_fake):
        # Calculate interpolation
        alpha = torch.rand(in_real.shape[0], 1, 1, 1, 1) if len(in_real.shape) == 5 else torch.rand(in_real.shape[0], 1, 1, 1)
        alpha = alpha.expand_as(in_real)
        alpha = alpha.cuda(in_real.device.index)
        interpolated = (alpha * in_real + (1 - alpha) * in_fake).requires_grad_(True)

        # Calculate probability of interpolated examples
        prob_interpolated = ref_disc(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).cuda(in_real.device.index), create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(in_real.shape[0], -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()
