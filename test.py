import torch
from vector_quantize_pytorch import ResidualVQ

residual_vq = ResidualVQ(
    dim = 256,
    num_quantizers = 8,      # specify number of quantizers
    codebook_size = 1024,    # codebook size
)

x = torch.randn(1, 1024, 256)

quantized, indices, commit_loss = residual_vq(x)
print(quantized.shape, indices.shape, commit_loss.shape)
# (1, 1024, 256), (1, 1024, 8), (1, 8)

# if you need all the codes across the quantization layers, just pass return_all_codes = True

quantized, indices, commit_loss, all_codes = residual_vq(x, return_all_codes = True)

# (8, 1, 1024, 256)