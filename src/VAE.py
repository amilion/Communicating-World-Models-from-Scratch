import torch
import torch.nn.functional as F
from torch import nn

class Encoder(nn.Module):
  def __init__(self, obs_dim, hidden_dim: int = 256, embed_dim: int = 256):
     super().__init__()
     self.net = nn.Sequential(
         nn.Linear(obs_dim, hidden_dim),
         nn.ReLU(),
         nn.Linear(hidden_dim, embed_dim)
     )

  def forward(self, obs):
     return self.net(obs)
 
class Decoder(nn.Module):
  def __init__(self,
               obs_dim,
               latent_dim : int = 32,
               hidden_dim : int = 128,
               hidden_layer : int = 256

  ):
     super().__init__()
     self.latent_dim = latent_dim
     self.hidden_dim = hidden_dim

     self.net = nn.Sequential(
         nn.Linear(latent_dim + hidden_dim, hidden_layer),
         nn.ReLU(),
         nn.Linear(hidden_layer, obs_dim)
     )
  def forward(self, z, h):
     return self.net(torch.cat([z, h], dim=-1))
