import torch
import torch.nn.functional as F

class AdversarialGame:
    def __init__(self, generator, discriminator):
        self.G = generator
        self.D = discriminator

    def value_function(self, x_real, z):
        """
        V(D, G) =
        E_x~pdata [ log D(x) ] +
        E_z~pz    [ log(1 - D(G(z))) ]
        """

        x_fake = self.G(z)

        d_real = self.D(x_real)
        d_fake = self.D(x_fake)

        term_real = torch.log(d_real + 1e-8)
        term_fake = torch.log(1 - d_fake + 1e-8)

        V = term_real.mean() + term_fake.mean()
        return V

    def generator_objective(self, z):
        """
        Non-saturating trick:
        maximize log D(G(z))
        """

        x_fake = self.G(z)
        d_fake = self.D(x_fake)

        return torch.log(d_fake + 1e-8).mean()
