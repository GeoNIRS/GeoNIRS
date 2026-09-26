from __future__ import annotations
from typing import Any
import torch
from torch import nn

def structured_contiguous_valid_patch_mask(
    valid: torch.Tensor,
    ratio: float,
    enabled: bool = True,
    return_span: bool = False,
):

    if valid.ndim != 3:
        raise ValueError("valid must be [B,C,P]")

    ratio = float(ratio)

    if not 0.0 <= ratio < 1.0:
        raise ValueError("temporal mask ratio must be in [0,1)")

    valid = valid.bool()

    B, C, P = valid.shape


    mask = torch.zeros_like(valid, dtype=torch.bool)


    span = torch.zeros(
        B,
        P,
        dtype=torch.bool,
        device=valid.device,
    )


    if not enabled or ratio == 0.0:
        return (mask, span) if return_span else mask

    for b in range(B):


        time_valid = valid[b].any(dim=0)


        valid_count = int(time_valid.sum().item())


        if valid_count <= 1:
            continue


        longest_run = 0
        current_run = 0

        for p in range(P):
            if time_valid[p]:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0

        if longest_run == 0:
            continue


        span_length = max(
            1,
            round(valid_count * ratio),
        )


        span_length = min(
            span_length,
            valid_count - 1,
        )


        span_length = min(
            span_length,
            longest_run,
        )


        candidate_starts = []

        for start in range(P - span_length + 1):
            end = start + span_length

            if time_valid[start:end].all():
                candidate_starts.append(start)

        if not candidate_starts:
            continue


        candidate_starts = torch.tensor(
            candidate_starts,
            device=valid.device,
            dtype=torch.long,
        )

        random_index = torch.randint(
            0,
            candidate_starts.numel(),
            (1,),
            device=valid.device,
        )

        start = int(candidate_starts[random_index].item())
        end = start + span_length


        span[b, start:end] = True



        mask[b, :, start:end] = valid[b, :, start:end]

    return (mask, span) if return_span else mask


class TemporalStreamEncoder(nn.Module):


    def __init__(
        self,
        patch_samples: int,
        max_patches: int,
        embedding_dim: int,
        num_heads: int,
        temporal_layers: int,
        mlp_ratio: float,
        dropout: float,
        time_init_std: float = 0.02,
    ):
        super().__init__()

        self.patch_samples = patch_samples
        self.max_patches = max_patches
        self.embedding_dim = embedding_dim


        self.patch_projection = nn.Linear(
            patch_samples,
            embedding_dim,
        )

        self.patch_norm = nn.LayerNorm(
            embedding_dim
        )


        self.time_embedding = nn.Embedding(
            max_patches,
            embedding_dim,
        )

        nn.init.normal_(
            self.time_embedding.weight,
            std=time_init_std,
        )

        ff_dim = round(
            embedding_dim * mlp_ratio
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )


        self.temporal_transformer = nn.TransformerEncoder(
            layer,
            num_layers=temporal_layers,
            norm=nn.LayerNorm(embedding_dim),
        )

        self.output_norm = nn.LayerNorm(
            embedding_dim
        )


    def patch_validity(
        self,
        sample_valid: torch.Tensor,
        minimum_patch_valid_fraction: float,
    ):

        valid_patches = sample_valid.bool().unfold(
            -1,
            self.patch_samples,
            self.patch_samples,
        )


        return (
            valid_patches.float().mean(dim=-1)
            >= minimum_patch_valid_fraction
        )


    def _patchify_signal(
        self,
        signal: torch.Tensor,
        sample_valid: torch.Tensor,
    ):


        signal = torch.where(
            sample_valid.bool(),
            signal,
            torch.zeros_like(signal),
        )

        return signal.unfold(
            -1,
            self.patch_samples,
            self.patch_samples,
        )


    def forward(
        self,
        signal: torch.Tensor,
        sample_valid: torch.Tensor,
        visible_valid: torch.Tensor,
    ):

        patches = self._patchify_signal(
            signal,
            sample_valid,
        )

        B, C, P, _ = patches.shape

        if P > self.max_patches:
            raise ValueError(
                f"P={P} exceeds max_patches={self.max_patches}"
            )





        tokens = self.patch_projection(
            patches
        )

        tokens = self.patch_norm(tokens)




        time_index = torch.arange(
            P,
            device=signal.device,
        )

        time_position = self.time_embedding(
            time_index
        ).view(
            1,
            1,
            P,
            self.embedding_dim,
        )

        tokens = tokens + time_position




        tokens = tokens * (
            visible_valid.unsqueeze(-1)
            .to(tokens.dtype)
        )







        temporal = tokens.reshape(
            B * C,
            P,
            self.embedding_dim,
        )

        temporal_valid = visible_valid.reshape(
            B * C,
            P,
        )







        active_rows = temporal_valid.any(dim=1)


        encoded = torch.zeros_like(
            temporal
        )




        if active_rows.any():

            active_temporal = temporal[
                active_rows
            ]

            active_valid = temporal_valid[
                active_rows
            ]



            padding_mask = ~active_valid

            active_temporal = (
                self.temporal_transformer(
                    active_temporal,
                    src_key_padding_mask=padding_mask,
                )
            )


            active_temporal = (
                active_temporal
                * active_valid.unsqueeze(-1)
                .to(active_temporal.dtype)
            )

            encoded[active_rows] = (
                active_temporal
            )




        encoded = encoded.reshape(
            B,
            C,
            P,
            self.embedding_dim,
        )

        encoded = self.output_norm(
            encoded
        )





        return encoded * (
            visible_valid.unsqueeze(-1)
            .to(encoded.dtype)
        )


class SourceDetectorGeometryEncoder(nn.Module):

    def __init__(
        self,
        geometry_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        spatial_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        if geometry_dim != 6:
            raise ValueError(
                "geometry_dim must be 6: [midpoint_x,midpoint_y,midpoint_z,delta_x,delta_y,delta_z]"
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.geometry_dim = geometry_dim
        self.embedding_dim = embedding_dim


        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )


        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.spatial_transformer = nn.TransformerEncoder(
            layer,
            num_layers=spatial_layers,
            norm=nn.LayerNorm(embedding_dim),
        )

    def forward(
        self,
        geometry: torch.Tensor,
        channel_valid: torch.Tensor,
    ):
        if geometry.ndim != 3 or geometry.shape[-1] != self.geometry_dim:
            raise ValueError(
                f"geometry must be [B,C,{self.geometry_dim}]"
            )

        if channel_valid.shape != geometry.shape[:2]:
            raise ValueError(
                "channel_valid must be [B,C]"
            )


        geometry = torch.where(
            channel_valid.unsqueeze(-1),
            geometry,
            torch.zeros_like(geometry),
        )


        tokens = self.geometry_projection(geometry)

        tokens = (
            tokens
            * channel_valid.unsqueeze(-1).to(tokens.dtype)
        )


        active_samples = channel_valid.any(dim=1)

        output = torch.zeros_like(tokens)

        if active_samples.any():

            active_tokens = tokens[active_samples]
            active_valid = channel_valid[active_samples]

            active_tokens = self.spatial_transformer(
                active_tokens,
                src_key_padding_mask=~active_valid,
            )

            active_tokens = (
                active_tokens
                * active_valid.unsqueeze(-1).to(active_tokens.dtype)
            )

            output[active_samples] = active_tokens

        return output


class BrainOmniChannelCompressor(nn.Module):
    

    def __init__(
        self,
        embedding_dim: int,
        num_latent_sources: int,
        num_heads: int,
        dropout: float,
        use_channel_geometry: bool = True,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.dim = embedding_dim
        self.num_latents = num_latent_sources
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads




        self.use_channel_geometry = bool(use_channel_geometry)


        self.latent_queries = nn.Parameter(
            torch.randn(num_latent_sources, embedding_dim) * 0.02
        )

        self.q_norm = nn.LayerNorm(embedding_dim)
        self.k_norm = nn.LayerNorm(embedding_dim)
        self.v_norm = nn.LayerNorm(embedding_dim)

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_norm = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def _heads(self, x):
        B, N, D = x.shape
        return x.view(
            B, N, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _merge(self, x):
        B, H, N, Dh = x.shape
        return x.transpose(1, 2).reshape(
            B, N, self.dim
        )

    def forward_pair(
        self,
        temporal_hbo,
        temporal_hbr,
        geometry_embedding,
        visible_valid,
        return_attention=False,
    ):
        B, C, P, D = temporal_hbo.shape


        hbo = temporal_hbo.permute(0, 2, 1, 3).reshape(B * P, C, D)
        hbr = temporal_hbr.permute(0, 2, 1, 3).reshape(B * P, C, D)

        valid = (
            visible_valid
            .permute(0, 2, 1)
            .reshape(B * P, C)
        )


        geometry = (
            geometry_embedding[:, None]
            .expand(B, P, C, D)
            .reshape(B * P, C, D)
        )


        q_input = (
            self.q_norm(self.latent_queries)
            .unsqueeze(0)
            .expand(B * P, -1, -1)
        )


        fused = 0.5 * (hbo + hbr)

        q = self._heads(
            self.q_proj(q_input)
        )

        k = self._heads(
            self.k_proj(
                self.k_norm(fused + geometry)
                if self.use_channel_geometry
                else self.k_norm(fused)
            )
        )





        scores = (
            q @ k.transpose(-2, -1)
        ) * (self.head_dim ** -0.5)


        active = valid.any(dim=1)

        weights = torch.zeros_like(scores)

        if active.any():
            active_scores = scores[active]

            active_scores = active_scores.masked_fill(
                ~valid[active, None, None, :],
                float("-inf"),
            )



            weights[active] = self.attn_dropout(
                torch.softmax(
                    active_scores,
                    dim=-1,
                )
            ).to(weights.dtype)





        def compress(signal):

            v = self._heads(
                self.v_proj(
                    self.v_norm(signal)
                )
            )

            x = weights @ v

            x = self.out_proj(
                self._merge(x)
            )

            x = self.out_norm(
                q_input + x
            )

            x = x + self.ffn(x)

            x[~active] = 0

            return (
                x.reshape(
                    B,
                    P,
                    self.num_latents,
                    D,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )

        output_hbo = compress(hbo)
        output_hbr = compress(hbr)

        attention = None

        if return_attention:
            attn = weights.reshape(
                B,
                P,
                self.num_heads,
                self.num_latents,
                C,
            )

            attention = {
                "per_head": attn,
                "mean": attn.mean(dim=2),
            }

        return output_hbo, output_hbr, attention


class ResidualProjectionBlock(nn.Module):


    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()


        self.skip = nn.Linear(input_dim, output_dim)


        self.main = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )


        self.residual_scale = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.residual_scale * self.main(x)


class FeatureProjection(nn.Module):

    def __init__(
        self,
        input_dim: int = 192,
        output_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim != 192 or output_dim != 64:
            raise ValueError(
                "Current projection is designed for 192 -> 128 -> 96 -> 64"
            )

        widths = [192, 128, 96, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.blocks = nn.ModuleList([
            ResidualProjectionBlock(
                widths[i],
                widths[i + 1],
                dropout,
            )
            for i in range(len(widths) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"input must be [B,L,P,{self.input_dim}]"
            )

        for block in self.blocks:
            x = block(x)

        return x

class CrossModalPhysicalDecoder(nn.Module):


    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int,
        patch_samples: int,
        max_patches: int,
        num_heads: int,
        temporal_layers: int,
        mlp_ratio: float,
        dropout: float,
        time_init_std: float = 0.02,
        use_channel_geometry: bool = True,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.latent_dim = int(latent_dim)
        self.embedding_dim = int(embedding_dim)
        self.patch_samples = int(patch_samples)
        self.max_patches = int(max_patches)





        self.use_channel_geometry = bool(use_channel_geometry)


        self.latent_up = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(
                self.latent_dim,
                self.embedding_dim,
            ),
        )


        self.channel_query_bias = nn.Parameter(
            torch.empty(self.embedding_dim)
        )
        nn.init.normal_(
            self.channel_query_bias,
            std=0.02,
        )





        self.decoder_mask_token = nn.Parameter(
            torch.empty(self.embedding_dim)
        )
        nn.init.normal_(
            self.decoder_mask_token,
            std=0.02,
        )


        self.time_embedding = nn.Embedding(
            self.max_patches,
            self.embedding_dim,
        )
        nn.init.normal_(
            self.time_embedding.weight,
            std=time_init_std,
        )


        self.query_norm = nn.LayerNorm(
            self.embedding_dim
        )
        self.key_norm = nn.LayerNorm(
            self.embedding_dim
        )
        self.value_norm = nn.LayerNorm(
            self.embedding_dim
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )


        self.post_cross = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(
                self.embedding_dim,
                self.embedding_dim * 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.embedding_dim * 2,
                self.embedding_dim,
            ),
        )


        ff_dim = round(
            self.embedding_dim * mlp_ratio
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_decoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=temporal_layers,
            norm=nn.LayerNorm(
                self.embedding_dim
            ),
        )


        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(
                self.embedding_dim,
                self.patch_samples,
            ),
        )

    def forward(
        self,
        compact_latent: torch.Tensor,
        geometry_embedding: torch.Tensor,
        full_valid: torch.Tensor,
        latent_time_valid: torch.Tensor,
    ) -> torch.Tensor:

        if compact_latent.ndim != 4:
            raise ValueError(
                "compact_latent must be [B,L,P,F]"
            )

        B, L, P, F = compact_latent.shape

        if F != self.latent_dim:
            raise ValueError(
                f"latent feature dim must be {self.latent_dim}"
            )

        if P > self.max_patches:
            raise ValueError(
                "patch count exceeds decoder max_patches"
            )

        C = full_valid.shape[1]

        if self.use_channel_geometry and geometry_embedding.shape != (
            B, C, self.embedding_dim
        ):
            raise ValueError(
                "geometry_embedding must be [B,C,D]"
            )

        if full_valid.shape != (B, C, P):
            raise ValueError(
                "full_valid must be [B,C,P]"
            )

        if latent_time_valid.shape != (B, P):
            raise ValueError(
                "latent_time_valid must be [B,P]"
            )






        latent = self.latent_up(
            compact_latent
        )



        latent = (
            latent
            .permute(0, 2, 1, 3)
            .reshape(
                B * P,
                L,
                self.embedding_dim,
            )
        )









        time_index = torch.arange(
            P,
            device=compact_latent.device,
        )

        time = self.time_embedding(
            time_index
        )

        if self.use_channel_geometry:

            query = (
                geometry_embedding[:, None, :, :]
                + time[None, :, None, :]
                + self.channel_query_bias.view(
                    1, 1, 1, -1
                )
            )

        else:


            query = (
                time[None, :, None, :]
                + self.channel_query_bias.view(
                    1, 1, 1, -1
                )
            )



            query = query.expand(
                B,
                P,
                C,
                self.embedding_dim,
            )


        query = query.reshape(
            B * P,
            C,
            self.embedding_dim,
        )












        physical = (
            query
            + self.decoder_mask_token.view(
                1, 1, -1
            )
        )


        active_time = (
            latent_time_valid
            .reshape(B * P)
        )





        if active_time.any():

            active_idx = (
                active_time
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )

            active_query = query[
                active_idx
            ]

            active_latent = latent[
                active_idx
            ]

            attended, _ = self.cross_attention(
                query=self.query_norm(
                    active_query
                ),
                key=self.key_norm(
                    active_latent
                ),
                value=self.value_norm(
                    active_latent
                ),
                need_weights=False,
            )

            active_physical = (
                active_query
                + attended
            )

            active_physical = (
                active_physical
                + self.post_cross(
                    active_physical
                )
            )



            physical = physical.index_copy(
                0,
                active_idx,
                active_physical,
            )








        physical = (
            physical
            .reshape(
                B,
                P,
                C,
                self.embedding_dim,
            )
            .permute(
                0, 2, 1, 3
            )
            .contiguous()
        )


        physical = (
            physical
            * full_valid
            .unsqueeze(-1)
            .to(physical.dtype)
        )








        temporal = physical.reshape(
            B * C,
            P,
            self.embedding_dim,
        )

        temporal_valid = (
            full_valid
            .reshape(B * C, P)
        )


        active_channels = (
            temporal_valid.any(dim=1)
        )

        decoded = torch.zeros_like(
            temporal
        )

        if active_channels.any():

            active_idx = (
                active_channels
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )

            x = temporal[
                active_idx
            ]

            valid = temporal_valid[
                active_idx
            ]

            x = self.temporal_decoder(
                x,
                src_key_padding_mask=~valid,
            )


            x = (
                x
                * valid.unsqueeze(-1)
                .to(x.dtype)
            )

            decoded = decoded.index_copy(
                0,
                active_idx,
                x,
            )




        decoded = decoded.reshape(
            B,
            C,
            P,
            self.embedding_dim,
        )






        prediction = self.reconstruction_head(
            decoded
        )

        return (
            prediction
            * full_valid
            .unsqueeze(-1)
            .to(prediction.dtype)
        )


class MultiScaleResidualReconstructionHead(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        patch_samples: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.patch_samples = int(patch_samples)
        self.input_norm = nn.LayerNorm(self.embedding_dim)
        self.feature_projection = nn.Linear(self.embedding_dim, hidden_dim)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in (3, 5, 7)
        )
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_norm = nn.LayerNorm(hidden_dim)
        self.residual_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.final_projection = nn.Linear(hidden_dim, self.patch_samples)
        self.sample_residual = nn.Linear(self.embedding_dim, self.patch_samples)

    def forward(self, decoded: torch.Tensor) -> torch.Tensor:
        if decoded.ndim != 4 or decoded.shape[-1] != self.embedding_dim:
            raise ValueError("decoded must be [B,C,P,embedding_dim]")
        batch, channels, patches, _ = decoded.shape
        normalized = self.input_norm(decoded)
        features = self.feature_projection(normalized)
        sequence = features.reshape(batch * channels, patches, -1).transpose(1, 2)
        multiscale = torch.cat([branch(sequence) for branch in self.branches], dim=1)
        sequence = sequence + self.feature_fusion(multiscale)
        residual_input = self.residual_norm(sequence.transpose(1, 2)).transpose(1, 2)
        sequence = sequence + self.residual_block(residual_input)
        fused = self.output_norm(sequence.transpose(1, 2)).reshape(batch, channels, patches, -1)
        return self.final_projection(fused) + self.sample_residual(normalized)

class _OursBase(nn.Module):

    def __init__(
        self,
        data_config: dict[str, Any],
        model_config: dict[str, Any],
    ) -> None:
        super().__init__()





        sampling_rate = float(
            data_config["sampling_rate_hz"]
        )

        patch_seconds = float(
            data_config["patch_seconds"]
        )

        window_seconds = float(
            data_config["window_seconds"]
        )

        self.patch_samples = round(
            sampling_rate * patch_seconds
        )

        window_samples = round(
            sampling_rate * window_seconds
        )

        if self.patch_samples <= 0:
            raise ValueError(
                "patch_samples must be positive"
            )

        if window_samples % self.patch_samples:
            raise ValueError(
                "window must be divisible by patch size"
            )

        self.patch_count = (
            window_samples
            // self.patch_samples
        )

        self.minimum_patch_valid_fraction = float(
            data_config["minimum_patch_valid_fraction"]
        )







        self.temporal_mask_ratio = float(
            model_config["temporal_mask_ratio"]
        )





        dim = int(
            model_config.get(
                "embedding_dim",
                192,
            )
        )

        heads = int(
            model_config.get(
                "num_heads",
                8,
            )
        )

        temporal_layers = int(
            model_config.get(
                "temporal_layers",
                4,
            )
        )

        mlp_ratio = float(
            model_config.get(
                "mlp_ratio",
                4.0,
            )
        )

        dropout = float(
            model_config.get(
                "dropout",
                0.1,
            )
        )

        num_latent_sources = int(
            model_config.get(
                "num_latent_sources",
                16,
            )
        )


        output_feature_dim = int(
            model_config.get(
                "output_feature_dim",
                64,
            )
        )

        self.embedding_dim = dim
        self.num_latent_sources = num_latent_sources
        self.output_feature_dim = output_feature_dim





        self.objective = str(
            model_config.get(
                "objective",
                "cross",
            )
        )

        if self.objective not in (
            "cross",
        ):
            raise ValueError(
                "objective must be cross, self, or hybrid"
            )




        self.use_channel_geometry = bool(
            model_config.get(
                "use_channel_geometry",
                True,
            )
        )





        encoder_kwargs = dict(
            patch_samples=self.patch_samples,
            max_patches=self.patch_count,
            embedding_dim=dim,
            num_heads=heads,
            temporal_layers=temporal_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            time_init_std=float(
                model_config.get(
                    "time_embedding_init_std",
                    0.02,
                )
            ),
        )

        self.hbo_encoder = TemporalStreamEncoder(
            **encoder_kwargs
        )

        self.hbr_encoder = TemporalStreamEncoder(
            **encoder_kwargs
        )











        self.geometry_encoder = (
            SourceDetectorGeometryEncoder(
                geometry_dim=6,
                embedding_dim=dim,
                hidden_dim=int(
                    model_config.get(
                        "geometry_hidden_dim",
                        dim,
                    )
                ),
                num_heads=int(
                    model_config.get(
                        "geometry_heads",
                        heads,
                    )
                ),
                spatial_layers=1,
                dropout=dropout,
            )
        )









        self.channel_compressor = (
            BrainOmniChannelCompressor(
                embedding_dim=dim,
                num_latent_sources=num_latent_sources,
                num_heads=int(
                    model_config.get(
                        "spatial_heads",
                        heads,
                    )
                ),
                dropout=dropout,
                use_channel_geometry=self.use_channel_geometry,
            )
        )







        self.hbo_projection = FeatureProjection(
            input_dim=dim,
            output_dim=output_feature_dim,
            dropout=dropout,
        )

        self.hbr_projection = FeatureProjection(
            input_dim=dim,
            output_dim=output_feature_dim,
            dropout=dropout,
        )









        decoder_kwargs = dict(
            latent_dim=output_feature_dim,
            embedding_dim=dim,
            patch_samples=self.patch_samples,
            max_patches=self.patch_count,

            num_heads=int(
                model_config.get(
                    "maev2_decoder_heads",
                    heads,
                )
            ),

            temporal_layers=int(
                model_config.get(
                    "maev2_decoder_temporal_layers",
                    1,
                )
            ),

            mlp_ratio=float(
                model_config.get(
                    "maev2_decoder_mlp_ratio",
                    2.0,
                )
            ),

            dropout=dropout,

            time_init_std=float(
                model_config.get(
                    "time_embedding_init_std",
                    0.02,
                )
            ),

            use_channel_geometry=self.use_channel_geometry,
        )


        if self.objective in (
            "cross",
        ):

            self.hbo_to_hbr_decoder = (
                CrossModalPhysicalDecoder(
                    **decoder_kwargs
                )
            )

            self.hbr_to_hbo_decoder = (
                CrossModalPhysicalDecoder(
                    **decoder_kwargs
                )
            )


        if self.objective in (
            "self",
            "hybrid",
        ):

            self.hbo_to_hbo_decoder = (
                CrossModalPhysicalDecoder(
                    **decoder_kwargs
                )
            )

            self.hbr_to_hbr_decoder = (
                CrossModalPhysicalDecoder(
                    **decoder_kwargs
                )
            )






    def _patch_valid(
        self,
        batch,
    ) -> torch.Tensor:

        patch_valid = (
            self.hbo_encoder.patch_validity(
                batch["valid"].bool(),
                self.minimum_patch_valid_fraction,
            )
        )


        return (
            patch_valid
            & batch["channel_valid"]
            .bool()
            .unsqueeze(-1)
        )









    def _prepare_signal_inputs(
        self,
        hbo,
        hbr,
        sample_valid,
    ):

        return (
            torch.where(
                sample_valid,
                hbo,
                torch.zeros_like(hbo),
            ),
            torch.where(
                sample_valid,
                hbr,
                torch.zeros_like(hbr),
            ),
        )






    def _patchify(
        self,
        value,
    ):

        return value.unfold(
            -1,
            self.patch_samples,
            self.patch_samples,
        )






    def _encode_pair(
        self,
        hbo,
        hbr,
        sample_valid,
        geometry_embedding,
        visible_valid,
        return_attention,
    ):






        temporal_hbo = self.hbo_encoder(
            hbo,
            sample_valid,
            visible_valid,
        )

        temporal_hbr = self.hbr_encoder(
            hbr,
            sample_valid,
            visible_valid,
        )









        (
            source_hbo,
            source_hbr,
            attention,
        ) = self.channel_compressor.forward_pair(
            temporal_hbo,
            temporal_hbr,
            geometry_embedding,
            visible_valid,
            return_attention=return_attention,
        )







        compact_hbo = self.hbo_projection(
            source_hbo
        )

        compact_hbr = self.hbr_projection(
            source_hbr
        )






        time_valid = visible_valid.any(
            dim=1
        )

        valid_scale = (
            time_valid[:, None, :, None]
            .to(compact_hbo.dtype)
        )

        compact_hbo = (
            compact_hbo
            * valid_scale
        )

        compact_hbr = (
            compact_hbr
            * valid_scale
        )

        return (
            compact_hbo,
            compact_hbr,
            source_hbo,
            source_hbr,
            attention,
        )






    def forward(
        self,
        batch,
        apply_mask=True,
        return_attention=False,
    ):

        hbo = batch["hbo"]
        hbr = batch["hbr"]
        geometry = batch["geometry"]

        sample_valid = batch.get(
            "valid_time_mask",
            batch.get("valid"),
        )

        channel_valid = batch.get(
            "valid_channel_mask",
            batch.get("channel_valid"),
        )

        if (
            sample_valid is None
            or channel_valid is None
        ):
            raise KeyError(
                "valid masks are required"
            )

        sample_valid = sample_valid.bool()
        channel_valid = channel_valid.bool()

        if (
            hbo.shape != hbr.shape
            or hbo.shape != sample_valid.shape
        ):
            raise ValueError(
                "hbo, hbr and valid must share [B,C,T]"
            )

        if (
            geometry.shape[:2] != hbo.shape[:2]
            or geometry.shape[-1] != 6
        ):
            raise ValueError(
                "geometry must be [B,C,6]"
            )





        hbo, hbr = self._prepare_signal_inputs(
            hbo,
            hbr,
            sample_valid,
        )





        local_batch = dict(
            batch,
            valid=sample_valid,
            channel_valid=channel_valid,
        )

        full_valid = self._patch_valid(
            local_batch
        )





        (
            temporal_mask,
            temporal_contiguous_span,
        ) = structured_contiguous_valid_patch_mask(
            full_valid,
            self.temporal_mask_ratio,
            enabled=bool(apply_mask),
            return_span=True,
        )

        source_mask = temporal_mask

        visible_valid = (
            full_valid
            & ~source_mask
        )









        geometry_embedding = (
            self.geometry_encoder(
                geometry,
                channel_valid,
            )
        )





        (
            compact_hbo,
            compact_hbr,
            source_hbo,
            source_hbr,
            shared_attention,
        ) = self._encode_pair(
            hbo,
            hbr,
            sample_valid,
            geometry_embedding,
            visible_valid,
            return_attention,
        )






        latent_time_valid = (
            visible_valid.any(dim=1)
        )





        pred_hbr_patches = None
        pred_hbo_patches = None

        pred_hbo_self_patches = None
        pred_hbr_self_patches = None


        if self.objective in (
            "cross",
            "hybrid",
        ):

            pred_hbr_patches = (
                self.hbo_to_hbr_decoder(
                    compact_hbo,
                    geometry_embedding,
                    full_valid,
                    latent_time_valid,
                )
            )

            pred_hbo_patches = (
                self.hbr_to_hbo_decoder(
                    compact_hbr,
                    geometry_embedding,
                    full_valid,
                    latent_time_valid,
                )
            )


        if self.objective in (
            "self",
            "hybrid",
        ):

            pred_hbo_self_patches = (
                self.hbo_to_hbo_decoder(
                    compact_hbo,
                    geometry_embedding,
                    full_valid,
                    latent_time_valid,
                )
            )

            pred_hbr_self_patches = (
                self.hbr_to_hbr_decoder(
                    compact_hbr,
                    geometry_embedding,
                    full_valid,
                    latent_time_valid,
                )
            )





        outputs = {

            "pred_hbr_patches":
                pred_hbr_patches,

            "pred_hbo_patches":
                pred_hbo_patches,

            "pred_hbo_self_patches":
                pred_hbo_self_patches,

            "pred_hbr_self_patches":
                pred_hbr_self_patches,

            "target_hbr_patches":
                self._patchify(hbr),

            "target_hbo_patches":
                self._patchify(hbo),

            "objective":
                self.objective,

            "sample_valid_patches":
                self._patchify(
                    sample_valid
                ),

            "mask":
                source_mask,

            "temporal_mask":
                temporal_mask,

            "temporal_contiguous_span":
                temporal_contiguous_span,

            "visible_valid":
                visible_valid,

            "valid":
                full_valid,



            "foundation_masked":
                torch.stack(
                    [
                        compact_hbo,
                        compact_hbr,
                    ],
                    dim=1,
                ),

            "latent_from_hbo":
                compact_hbo,

            "latent_from_hbr":
                compact_hbr,


            "latent_from_hbo_full":
                source_hbo,

            "latent_from_hbr_full":
                source_hbr,
        }

        if return_attention:

            outputs[
                "attention_shared"
            ] = shared_attention["mean"]

            outputs[
                "attention_shared_per_head"
            ] = shared_attention["per_head"]

        return outputs









    def encode(
        self,
        batch,
        return_attention=False,
    ):

        hbo = batch["hbo"]
        hbr = batch["hbr"]

        sample_valid = batch.get(
            "valid_time_mask",
            batch.get("valid"),
        ).bool()

        channel_valid = batch.get(
            "valid_channel_mask",
            batch.get("channel_valid"),
        ).bool()

        hbo, hbr = self._prepare_signal_inputs(
            hbo,
            hbr,
            sample_valid,
        )

        local_batch = dict(
            batch,
            valid=sample_valid,
            channel_valid=channel_valid,
        )

        full_valid = self._patch_valid(
            local_batch
        )

        geometry_embedding = (
            self.geometry_encoder(
                batch["geometry"],
                channel_valid,
            )
        )

        (
            compact_hbo,
            compact_hbr,
            source_hbo,
            source_hbr,
            attention,
        ) = self._encode_pair(
            hbo,
            hbr,
            sample_valid,
            geometry_embedding,
            full_valid,
            return_attention,
        )







        foundation = torch.stack(
            [
                compact_hbo,
                compact_hbr,
            ],
            dim=1,
        )



        joint_concat = torch.cat(
            [
                compact_hbo,
                compact_hbr,
            ],
            dim=-1,
        )

        time_valid = full_valid.any(
            dim=1
        )

        latent_valid = (
            time_valid[:, None, :]
            .expand(
                time_valid.shape[0],
                self.num_latent_sources,
                time_valid.shape[1],
            )
        )

        result = {

            "latent_hbo":
                compact_hbo,

            "latent_hbr":
                compact_hbr,

            "latent_hbo_full":
                source_hbo,

            "latent_hbr_full":
                source_hbr,

            "joint_concat":
                joint_concat,

            "foundation":
                foundation,

            "valid":
                latent_valid,

            "physical_valid":
                full_valid,

            "geometry_embedding":
                geometry_embedding,
        }

        if return_attention:

            result[
                "attention_shared"
            ] = attention["mean"]

            result[
                "attention_shared_per_head"
            ] = attention["per_head"]

        return result


class Ours(_OursBase):
    def __init__(self, data_config: dict[str, Any], model_config: dict[str, Any]) -> None:
        super().__init__(data_config, model_config)
        if self.objective != "cross": raise ValueError("canonical Ours requires objective='cross'")
        hidden = int(model_config.get("new_decoder_head_hidden_dim", 128))
        dropout = float(model_config.get("new_decoder_head_dropout", model_config.get("dropout", 0.1)))
        for decoder in (self.hbo_to_hbr_decoder, self.hbr_to_hbo_decoder):
            decoder.reconstruction_head = MultiScaleResidualReconstructionHead(decoder.embedding_dim, decoder.patch_samples, hidden, dropout)

    def forward(self, batch, apply_mask=True, return_attention=False):
        if self.objective != "cross": raise ValueError("the canonical asymmetric model requires objective='cross'")
        hbo, hbr, geometry = batch["hbo"], batch["hbr"], batch["geometry"]
        sample_valid = batch.get("valid_time_mask", batch.get("valid")); channel_valid = batch.get("valid_channel_mask", batch.get("channel_valid"))
        if sample_valid is None or channel_valid is None: raise KeyError("valid masks are required")
        sample_valid, channel_valid = sample_valid.bool(), channel_valid.bool()
        if hbo.shape != hbr.shape or hbo.shape != sample_valid.shape: raise ValueError("hbo, hbr and valid must share [B,C,T]")
        if geometry.shape[:2] != hbo.shape[:2] or geometry.shape[-1] != 6: raise ValueError("geometry must be [B,C,6]")
        hbo, hbr = self._prepare_signal_inputs(hbo, hbr, sample_valid)
        full_valid = self._patch_valid(dict(batch, valid=sample_valid, channel_valid=channel_valid))
        target_mask, temporal_contiguous_span = structured_contiguous_valid_patch_mask(full_valid, self.temporal_mask_ratio, enabled=bool(apply_mask), return_span=True)
        target_visible_valid, source_visible_valid = full_valid & ~target_mask, full_valid
        geometry_embedding = self.geometry_encoder(geometry, channel_valid)
        temporal_hbo_source = self.hbo_encoder(hbo, sample_valid, source_visible_valid)
        temporal_hbr_target = self.hbr_encoder(hbr, sample_valid, target_visible_valid)
        source_hbo, _, attention_hbo = self.channel_compressor.forward_pair(temporal_hbo_source, temporal_hbr_target, geometry_embedding, source_visible_valid, return_attention=return_attention)
        compact_hbo = self.hbo_projection(source_hbo)
        temporal_hbo_target = self.hbo_encoder(hbo, sample_valid, target_visible_valid)
        temporal_hbr_source = self.hbr_encoder(hbr, sample_valid, source_visible_valid)
        _, source_hbr, attention_hbr = self.channel_compressor.forward_pair(temporal_hbo_target, temporal_hbr_source, geometry_embedding, source_visible_valid, return_attention=return_attention)
        compact_hbr = self.hbr_projection(source_hbr)
        time_valid = source_visible_valid.any(dim=1); valid_scale = time_valid[:, None, :, None].to(compact_hbo.dtype)
        compact_hbo, compact_hbr = compact_hbo * valid_scale, compact_hbr * valid_scale
        pred_hbr_patches = self.hbo_to_hbr_decoder(compact_hbo, geometry_embedding, full_valid, time_valid)
        pred_hbo_patches = self.hbr_to_hbo_decoder(compact_hbr, geometry_embedding, full_valid, time_valid)
        outputs = {"pred_hbr_patches": pred_hbr_patches, "pred_hbo_patches": pred_hbo_patches, "pred_hbo_self_patches": None, "pred_hbr_self_patches": None, "target_hbr_patches": self._patchify(hbr), "target_hbo_patches": self._patchify(hbo), "objective": self.objective, "sample_valid_patches": self._patchify(sample_valid), "mask": target_mask, "temporal_mask": target_mask, "temporal_contiguous_span": temporal_contiguous_span, "visible_valid": source_visible_valid, "source_visible_valid": source_visible_valid, "target_visible_valid": target_visible_valid, "valid": full_valid, "masking_strategy": "asymmetric_cross_modal", "foundation_masked": torch.stack([compact_hbo, compact_hbr], dim=1), "latent_from_hbo": compact_hbo, "latent_from_hbr": compact_hbr, "latent_from_hbo_full": source_hbo, "latent_from_hbr_full": source_hbr, "temporal_source_hbo": temporal_hbo_source, "temporal_source_hbr": temporal_hbr_source, "temporal_target_hbo": temporal_hbo_target, "temporal_target_hbr": temporal_hbr_target}
        if return_attention:
            outputs["attention_hbo_source"], outputs["attention_hbr_source"] = attention_hbo["mean"], attention_hbr["mean"]
            outputs["attention_hbo_source_per_head"], outputs["attention_hbr_source_per_head"] = attention_hbo["per_head"], attention_hbr["per_head"]
            outputs["attention_shared"] = 0.5 * (attention_hbo["mean"] + attention_hbr["mean"])
        return outputs







def bidirectional_cross_mae_loss(
    outputs: dict[str, torch.Tensor],
    sample_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
   
    patch_mask = (
        outputs["mask"]
        .bool()
        .unsqueeze(-1)
    )

    sample_valid = (
        outputs["sample_valid_patches"]
        .bool()
    )


    selected = (
        patch_mask
        & sample_valid
    )





    pred_hbr = (
        outputs["pred_hbr_patches"]
        .float()
    )

    target_hbr = (
        outputs["target_hbr_patches"]
        .float()
    )

    pred_hbo = (
        outputs["pred_hbo_patches"]
        .float()
    )

    target_hbo = (
        outputs["target_hbo_patches"]
        .float()
    )

    if pred_hbr.shape != target_hbr.shape:
        raise ValueError(
            "pred_hbr_patches and target_hbr_patches "
            "must have the same shape"
        )

    if pred_hbo.shape != target_hbo.shape:
        raise ValueError(
            "pred_hbo_patches and target_hbo_patches "
            "must have the same shape"
        )

    if selected.shape != target_hbo.shape:
        raise ValueError(
            "mask/sample_valid shape must match patch targets"
        )

    B, C, P, S = selected.shape

    N = P * S





    if sample_weights is not None:

        if sample_weights.shape != (B,):
            raise ValueError(
                "sample_weights must have shape [B]"
            )

        sample_weights = (
            sample_weights
            .float()
            .to(pred_hbo.device)
        )


        bc_weights = (
            sample_weights[:, None]
            .expand(B, C)
            .reshape(B * C)
        )

    else:

        bc_weights = None





    def aggregate(
        value: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:

        active = active.bool()

        if bc_weights is None:

            weight = active.to(
                value.dtype
            )

        else:

            weight = (
                bc_weights.to(value.dtype)
                * active.to(value.dtype)
            )

        weight_sum = (
            weight.sum()
            .clamp_min(1e-8)
        )

        return (
            (value * weight).sum()
            / weight_sum
        )





    def direction_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:









        pred_flat = pred.reshape(
            B * C,
            N,
        )

        target_flat = target.reshape(
            B * C,
            N,
        )

        mask_flat = selected.reshape(
            B * C,
            N,
        )

        mask_float = mask_flat.to(
            pred_flat.dtype
        )


        selected_count = (
            mask_flat.sum(dim=-1)
        )





        active_time = (
            selected_count > 0
        )

        time_per_channel = (
            (
                (pred_flat - target_flat)
                .abs()
                * mask_float
            )
            .sum(dim=-1)
            /
            selected_count
            .clamp_min(1)
            .to(pred_flat.dtype)
        )

        time_loss = aggregate(
            time_per_channel,
            active_time,
        )







        count_safe = (
            selected_count
            .clamp_min(1)
            .to(pred_flat.dtype)
        )

        pred_mean = (
            (pred_flat * mask_float)
            .sum(dim=-1)
            / count_safe
        )

        target_mean = (
            (target_flat * mask_float)
            .sum(dim=-1)
            / count_safe
        )

        pred_centered = (
            (
                pred_flat
                - pred_mean.unsqueeze(-1)
            )
            * mask_float
        )

        target_centered = (
            (
                target_flat
                - target_mean.unsqueeze(-1)
            )
            * mask_float
        )

        pred_energy = (
            pred_centered
            .pow(2)
            .sum(dim=-1)
        )

        target_energy = (
            target_centered
            .pow(2)
            .sum(dim=-1)
        )

        covariance = (
            pred_centered
            * target_centered
        ).sum(dim=-1)

        eps = 1e-8

        pcc = (
            covariance
            /
            torch.sqrt(
                pred_energy
                * target_energy
                + eps
            )
        )

        pcc = torch.clamp(
            pcc,
            min=-1.0,
            max=1.0,
        )







        valid_pcc = (
            (pred_energy > eps)
            & (target_energy > eps)
        )

        pcc_per_channel = torch.where(
            valid_pcc,
            torch.exp(-pcc),
            torch.ones_like(pcc),
        )


        spectral_active = (
            selected_count >= 2
        )

        pcc_loss = aggregate(
            pcc_per_channel,
            spectral_active,
        )




























        compact_rank = (
            mask_flat
            .long()
            .cumsum(dim=-1)
            - 1
        )

        compact_rank = (
            compact_rank
            .clamp_min(0)
        )

        compact_pred = (
            torch.zeros_like(pred_flat)
            .scatter_add(
                dim=1,
                index=compact_rank,
                src=pred_flat * mask_float,
            )
        )

        compact_target = (
            torch.zeros_like(target_flat)
            .scatter_add(
                dim=1,
                index=compact_rank,
                src=target_flat * mask_float,
            )
        )













        amp_per_channel = torch.zeros(
            B * C,
            device=pred.device,
            dtype=pred.dtype,
        )

        phase_per_channel = torch.zeros_like(
            amp_per_channel
        )








        valid_lengths = torch.unique(
            selected_count[
                spectral_active
            ]
        ).tolist()

        for valid_length in valid_lengths:

            k = int(valid_length)

            rows = torch.nonzero(
                selected_count == k,
                as_tuple=False,
            ).squeeze(-1)

            if rows.numel() == 0:
                continue


            pred_group = (
                compact_pred
                .index_select(
                    0,
                    rows,
                )[:, :k]
            )

            target_group = (
                compact_target
                .index_select(
                    0,
                    rows,
                )[:, :k]
            )





            pred_fft = torch.fft.rfft(
                pred_group,
                dim=-1,
                norm="ortho",
            )

            target_fft = torch.fft.rfft(
                target_group,
                dim=-1,
                norm="ortho",
            )





            pred_amp = torch.abs(
                pred_fft
            )

            target_amp = torch.abs(
                target_fft
            )

            amp_group = (
                (
                    pred_amp
                    - target_amp
                )
                .abs()
                .mean(dim=-1)
            )








            pred_phase = torch.angle(
                pred_fft
            )

            target_phase = torch.angle(
                target_fft
            )

            phase_diff = (
                pred_phase
                - target_phase
            )

            wrapped_phase_diff = torch.atan2(
                torch.sin(
                    phase_diff
                ),
                torch.cos(
                    phase_diff
                ),
            )

            phase_group = (
                wrapped_phase_diff
                .abs()
                .mean(dim=-1)
            )







            amp_per_channel = (
                amp_per_channel
                .index_copy(
                    0,
                    rows,
                    amp_group,
                )
            )

            phase_per_channel = (
                phase_per_channel
                .index_copy(
                    0,
                    rows,
                    phase_group,
                )
            )





        amp_loss = aggregate(
            amp_per_channel,
            spectral_active,
        )

        phase_loss = aggregate(
            phase_per_channel,
            spectral_active,
        )





        total = (
            time_loss
            + pcc_loss
            + amp_loss
            + 0.5 * phase_loss
        )

        return {
            "total": total,
            "time": time_loss,
            "pcc": pcc_loss,
            "amp": amp_loss,
            "phase": phase_loss,
        }





    hbo_to_hbr = direction_loss(
        pred_hbr,
        target_hbr,
    )





    hbr_to_hbo = direction_loss(
        pred_hbo,
        target_hbo,
    )





    cross_loss = 0.5 * (
        hbo_to_hbr["total"]
        + hbr_to_hbo["total"]
    )





    return {
        "loss":
            cross_loss,

        "total_loss":
            cross_loss,

        "cross_loss":
            cross_loss,





        "hbo_to_hbr_loss":
            hbo_to_hbr["total"],

        "hbo_to_hbr_time_loss":
            hbo_to_hbr["time"],

        "hbo_to_hbr_pcc_loss":
            hbo_to_hbr["pcc"],

        "hbo_to_hbr_amp_loss":
            hbo_to_hbr["amp"],

        "hbo_to_hbr_phase_loss":
            hbo_to_hbr["phase"],





        "hbr_to_hbo_loss":
            hbr_to_hbo["total"],

        "hbr_to_hbo_time_loss":
            hbr_to_hbo["time"],

        "hbr_to_hbo_pcc_loss":
            hbr_to_hbo["pcc"],

        "hbr_to_hbo_amp_loss":
            hbr_to_hbo["amp"],

        "hbr_to_hbo_phase_loss":
            hbr_to_hbo["phase"],





        "masked_sample_count":
            selected.sum(),
    }


__all__ = ["Ours", "bidirectional_cross_mae_loss", "TemporalStreamEncoder", "SourceDetectorGeometryEncoder", "BrainOmniChannelCompressor", "ResidualProjectionBlock", "FeatureProjection", "CrossModalPhysicalDecoder", "MultiScaleResidualReconstructionHead", "structured_contiguous_valid_patch_mask"]
