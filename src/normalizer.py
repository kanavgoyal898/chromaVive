import torch
import torch.nn as nn

class Normalizer(nn.Module):
    """
    Normalizer class to normalize and denormalize the L and AB channels
    in the LAB color space.

    Attributes:
        l_norm (float): Normalization factor for the L (lightness) channel.
        ab_norm (float): Normalization factor for the AB (color) channels.
        l_centeralize (bool): Whether to centralize the L channel around l_norm/2.
        ab_centeralize (bool): Whether to centralize the AB channels around ab_norm/2.
        l_center (float): Central value for the L channel, calculated based on l_norm.
        ab_center (float): Central value for the AB channels, calculated based on ab_norm.
    """

    def __init__(self, l_norm=100., ab_norm=110., l_centeralize=True, ab_centeralize=False):
        super().__init__()

        self.l_norm = l_norm
        self.ab_norm = ab_norm

        self.l_centeralize = l_centeralize
        self.ab_centeralize = ab_centeralize

        self.l_center = self.l_norm / 2 if self.l_centeralize else 0
        self.ab_center = self.ab_norm / 2 if self.ab_centeralize else 0

    def normalize_l(self, l_in):
        """
        Normalize the L channel
        """
        return (l_in - self.l_center) / self.l_norm
    
    def normalize_ab(self, ab_in):
        """
        Normalize the AB channel
        """
        return (ab_in - self.ab_center) / self.ab_norm
    
    def denormalize_l(self, l_in):
        """
        Denormalize the L channel"""
        return l_in * self.l_norm + self.l_center
    
    def denormalize_ab(self, ab_in):
        """
        Denormalize the AB channel
        """
        return ab_in * self.ab_norm + self.ab_center