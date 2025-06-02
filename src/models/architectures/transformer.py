import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan import KANLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Standard positional encoding with Sin/Cos functions + LayerNorm to preserve 
        temporal relationships between frames throughtout sequence-modeling.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Precompute positional encodings (PE) using sinusoidal functions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)
        self.norm_pe = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor with positional encodings added and normalized
        """
        seq_len = x.size(0)
        x2 = x + self.pe[:seq_len, :]  # Add positional encodings
        x2 = self.norm_pe(x2)          # Normalize
        return self.dropout(x2)


class Encoder_TRANSFORMER(nn.Module):
    """
    Encoder module using Transformer architecture with KAN layers.
    Key components:
    - KANLayer which eplaces linear projections with learnable 1D splines;
    - Transformer Encoder processing temporal dependencies.
        """
    def __init__(
        self,
        modeltype,
        njoints: int,
        nfeats: int,
        num_frames: int,
        num_classes: int,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim: int = 256,              
        ff_size: int = 1024,                
        num_layers: int = 4,                
        num_heads: int = 4,                 
        dropout: float = 0.1,
        activation: str = "gelu",
        **kargs
    ):
        super().__init__()
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim                            # Latent space dimensionality
        self.ff_size = ff_size                                  # Feedforward network size
        self.num_layers = num_layers                            # Transformer layers
        self.num_heads = num_heads                              # Multi-head attention heads
        self.dropout = dropout
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats           # Input feature dimension

        # Learnable parameters for μ and σ (variational posterior)
        self.muQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, self.latent_dim))

        # KANLayer for skeleton embedding:
        # Input: njoints * nfeats (flattened joint features)
        # Output: latent_dim (compressed representation)
        # KANLayer replaces linear projections with a matrix of 1D B-splines
        self.skelEmbedding = KANLayer(self.input_feats, self.latent_dim)

        # Positional Encoding for temporal alignment
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # Transformer Encoder with multi-head attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.seqTransEncoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.encoder_norm = nn.LayerNorm(self.latent_dim)  # Final normalization

    def forward(self, batch: dict) -> dict:
        """
        batch["x"]: (batch, njoints, nfeats, nframes)
        batch["y"]: (batch,)  — classes (if none, then == 0)
        batch["mask"]: (batch, nframes) — bool-mask of actual frames
        """
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, nj, nf, nf2 = x.shape  # nf2 = nframes
        assert nf2 == self.num_frames, "Frame dimension mismatch"

        # Reshape input: (nframes, batch, njoints*nfeats)
        x_seq = x.permute(3, 0, 1, 2).reshape(self.num_frames, bs, self.input_feats)

        # Applies learnable 1D splines to input features
        x_emb = self.skelEmbedding(x_seq)  # (nframes, batch, latent_dim)

        # Handle class labels (y)
        if y is None:
            y = torch.zeros(bs, dtype=torch.long, device=x.device)
        else:
            y = y.clamp(0, self.num_classes - 1)

        # Initialize mu and sigma queries:
        mu_init = self.muQuery.expand(bs, -1)       # (batch, latent_dim)
        sigma_init = self.sigmaQuery.expand(bs, -1) # (batch, latent_dim)

        # Concatenate [mu, sigma, x_emb] for Transformer input
        mu_init = mu_init.unsqueeze(0)       # (1, batch, latent_dim)
        sigma_init = sigma_init.unsqueeze(0) # (1, batch, latent_dim)
        xcat = torch.cat((mu_init, sigma_init, x_emb), dim=0)  # (2 + nframes, batch, latent_dim)

        # Update mask for mu/sigma
        mu_sigma_mask = torch.ones((bs, 2), dtype=torch.bool, device=x.device)
        mask_seq = torch.cat((mu_sigma_mask, mask), dim=1)  # (batch, 2 + nframes)

        # Add positional encodings
        xcat_pe = self.sequence_pos_encoder(xcat)  # (2 + nframes, batch, latent_dim)

        # Transformer Encoder
        encoded = self.seqTransEncoder(
            xcat_pe,
            src_key_padding_mask=~mask_seq  # True = mask padding
        )  # (2 + nframes, batch, latent_dim)

        # Final normalization
        encoded = self.encoder_norm(encoded)

        # Extract mu and logvar (logvar stors in encoded)
        mu = encoded[0]      # (batch, latent_dim)
        logvar = encoded[1]  # (batch, latent_dim)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # (batch, latent_dim)

        return {"mu": mu, "logvar": logvar, "z": z}


class Decoder_TRANSFORMER(nn.Module):
    """
    Decoder module using Transformer architecture with KAN-layer:
    - KANLayer: Final projection layer for skeleton reconstruction
    - Transformer Decoder: Autoregressive generation of sequences
    """
    def __init__(
        self,
        modeltype,
        njoints: int,
        nfeats: int,
        num_frames: int,
        num_classes: int,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kargs
    ):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        # Bias parameters for action-specific generation
        self.actionBiases = nn.Parameter(torch.randn(1, self.latent_dim))

        # Positional Encoding for temporal queries
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.seqTransDecoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
        self.decoder_norm = nn.LayerNorm(self.latent_dim)  # Final normalization

        # Final KANLayer for skeleton reconstruction:
        # Input: latent_dim
        # Output: input_feats (reconstructed joint features)
        self.finallayer = KANLayer(self.latent_dim, self.input_feats)

    def forward(self, batch: dict, use_text_emb: bool = False) -> dict:
        """
        Forward pass for the decoder.
        Args:
            batch: Dictionary containing latent codes and metadata
            use_text_emb: Whether to use text embeddings instead of latent codes
        Returns:
            Dictionary with generated output
        """
        z = batch["z"]  # Latent code: (batch, latent_dim)
        y = batch["y"]
        mask = batch["mask"]  # (batch, nframes)
        lengths = batch.get("lengths", None)
        bs, nframes = mask.shape
        nj, nf = self.njoints, self.nfeats

        # Use text embeddings if specified
        if use_text_emb:
            z = batch["clip_text_emb"]  # (batch, latent_dim)

        # Normalize latent code
        z = F.layer_norm(z, (self.latent_dim,))  # (batch, latent_dim)
        z = z.unsqueeze(0)  # (1, batch, latent_dim) — memory for decoder

        # Generate time queries: (nframes, batch, latent_dim)
        timequeries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        # Add positional encodings
        timequeries_pe = self.sequence_pos_encoder(timequeries)

        # Ensure mask is boolean
        if mask.dtype != torch.bool:
            mask = mask.bool()

        # Transformer Decoder
        dec_out = self.seqTransDecoder(
            tgt=timequeries_pe,
            memory=z,
            tgt_key_padding_mask=~mask  
        )  # (nframes, batch, latent_dim)

        # Final normalization of the output of decoder
        dec_out = self.decoder_norm(dec_out)  # (nframes, batch, latent_dim)

        # Transforming decoder output via KANLayer into skeletal features (reconstruct)
        skel_feats = self.finallayer(dec_out)  # (nframes, batch, input_feats)
        skel_feats = skel_feats.view(nframes, bs, nj, nf)  # (nframes, batch, njoints, nfeats) --> Reshape to joints

        # Apply mask to zero out padding
        mask_t = mask.T  # (nframes, batch)
        skel_feats[~mask_t] = 0.0

        # Final output format: (batch, njoints, nfeats, nframes)
        output = skel_feats.permute(1, 2, 3, 0).contiguous()

        if use_text_emb:
            batch["txt_output"] = output
        else:
            batch["output"] = output

        return batch