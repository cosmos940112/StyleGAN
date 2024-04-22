import torch
import torch.nn as nn
from stylegan2_pytorch import Generator, Discriminator

class ConditionalGenerator(Generator):
    def __init__(self, image_size, latent_dim, num_classes, network_capacity=16, transparent=False, attn_layers=None, no_const=False, fmap_max=512):
        if attn_layers is None:
            attn_layers = []
        super().__init__(image_size=image_size, latent_dim=latent_dim, network_capacity=network_capacity, transparent=transparent, attn_layers=attn_layers, no_const=no_const, fmap_max=fmap_max)
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, latent_dim)

    def forward(self, input, labels, input_noise):
        class_embed = self.class_embedding(labels).expand(-1, self.latent_dim)  # Ensure class embedding matches the latent dimension
        mixed_input = input + class_embed  # Combine input latent with class embedding
        return super().forward(mixed_input, input_noise)

class ConditionalDiscriminator(Discriminator):
    def __init__(self, image_size, num_classes, network_capacity=16, transparent=False, attn_layers=None, fq_layers=None, fq_dict_size=256, fmap_max=512):
        if attn_layers is None:
            attn_layers = []
        if fq_layers is None:
            fq_layers = []
        super().__init__(image_size=image_size, network_capacity=network_capacity, fq_layers=fq_layers, fq_dict_size=fq_dict_size, attn_layers=attn_layers, transparent=transparent, fmap_max=fmap_max)
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, image_size * image_size)

    def forward(self, input, labels):
        class_embed = self.class_embedding(labels).view(labels.size(0), 1, int(torch.sqrt(self.class_embedding.embedding_dim)), int(torch.sqrt(self.class_embedding.embedding_dim)))
        combined_input = torch.cat([input, class_embed], dim=1)
        return super().forward(combined_input)
