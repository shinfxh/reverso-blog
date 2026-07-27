"""
Reverso (torch-native): conv-attention hybrid for time series forecasting.

Drop-in replacement for reverso.model that uses only pure PyTorch operations.
No FlashFFTConv or flash-linear-attention (fla) required.
Checkpoint-compatible with the original model.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=6500):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class RMSNorm(nn.Module):
    """Minimal RMSNorm matching fla.modules.layernorm.RMSNorm weight layout."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms * self.weight.float()).to(dtype)


# ---------------------------------------------------------------------------
# Gating & MLP  (identical to reverso.model)
# ---------------------------------------------------------------------------

class Gating(nn.Module):
    def __init__(self, channels, temporal_kernel=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=temporal_kernel,
                      padding=temporal_kernel // 2, groups=channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class MLPBlock(nn.Module):
    def __init__(self, d_in, d_out, d_intermediate=0):
        super().__init__()
        self.norm = nn.LayerNorm(d_out)
        if d_intermediate and d_intermediate > 0:
            self.linear = nn.Linear(d_in, d_intermediate)
            self.linear_final = nn.Linear(d_intermediate, d_out)
        else:
            self.linear = nn.Linear(d_in, d_out)
            self.linear_final = nn.Identity()
        self.activation = nn.ReLU()
        self.skip_linear = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        if x.ndim == 3:
            x = x.permute(0, 2, 1)
        residual = self.skip_linear(x)
        y = self.linear(x)
        y = self.activation(y)
        y = self.linear_final(y)
        y = self.norm(y)
        y = residual + y
        if y.ndim == 3:
            y = y.permute(0, 2, 1)
        return y


# ---------------------------------------------------------------------------
# CNNBlock — replaces FlashFFTConv with torch.fft circular convolution
# ---------------------------------------------------------------------------

class CNNBlock(nn.Module):
    def __init__(self, channels, seq_len, gating_kernel_size=3):
        super().__init__()
        self.seq_len = seq_len
        self.k = nn.Parameter(torch.randn(channels, seq_len, dtype=torch.float32))
        self.pregate = Gating(channels, gating_kernel_size)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        residual = x
        # Match original precision: quantise to bf16 then back to f32,
        # so the FFT sees the same rounded values the model was trained with.
        x_conv = x.contiguous().to(torch.bfloat16)
        pregate = self.pregate(x_conv.float()).to(x_conv.dtype)
        x_gated = (pregate * x_conv).float()

        # Circular convolution via FFT (matches FlashFFTConv behaviour)
        X = torch.fft.rfft(x_gated, n=self.seq_len, dim=-1)
        K = torch.fft.rfft(self.k.float(), n=self.seq_len, dim=-1)
        out = torch.fft.irfft(X * K.unsqueeze(0), n=self.seq_len, dim=-1)

        out = self.activation(out)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        out = out + residual
        return out

    def forward_browser(self, x):
        """Export-friendly equivalent using WebGPU-supported ONNX operators.

        ONNX Runtime WebGPU does not currently provide the ONNX DFT operator.
        A circular depthwise Conv1d is mathematically equivalent to the FFT
        convolution above and exports as Pad/Concat/Conv operations. Browser
        inference uses float32 because WebGPU does not expose bfloat16.
        """
        residual = x
        x_conv = x.contiguous().float()
        pregate = self.pregate(x_conv)
        x_gated = pregate * x_conv

        circular_input = torch.cat(
            [x_gated, x_gated[..., : self.seq_len - 1]],
            dim=-1,
        )
        circular_kernel = torch.cat(
            [self.k[:, :1], self.k[:, 1:].flip(-1)],
            dim=-1,
        ).unsqueeze(1)
        out = F.conv1d(
            circular_input,
            circular_kernel,
            groups=x_gated.shape[1],
        )

        out = self.activation(out)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        return out + residual


# ---------------------------------------------------------------------------
# TorchDeltaNet — pure-PyTorch delta-rule linear attention
# Weight-compatible with fla.layers.DeltaNet (same parameter names & shapes)
# ---------------------------------------------------------------------------

class TorchDeltaNet(nn.Module):
    """
    Pure-PyTorch implementation of DeltaNet (delta-rule linear attention).

    Supports the same constructor kwargs and produces identical state-dict keys
    as ``fla.layers.DeltaNet`` so that pre-trained checkpoints load directly.
    """

    def __init__(
        self,
        d_model: int = None,
        hidden_size: int = 1024,
        mode: str = 'chunk',
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # --- projections (match fla naming) ---
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            # fla's ShortConvolution inherits nn.Conv1d(groups=D, padding=W-1)
            self.q_conv1d = nn.Conv1d(
                self.key_dim, self.key_dim, conv_size,
                padding=conv_size - 1, groups=self.key_dim, bias=conv_bias,
            )
            self.k_conv1d = nn.Conv1d(
                self.key_dim, self.key_dim, conv_size,
                padding=conv_size - 1, groups=self.key_dim, bias=conv_bias,
            )
            self.v_conv1d = nn.Conv1d(
                self.value_dim, self.value_dim, conv_size,
                padding=conv_size - 1, groups=self.value_dim, bias=conv_bias,
            )

        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ---- causal short convolution (pure torch) ----------------------------

    def _causal_conv1d(self, x: torch.Tensor, conv: nn.Conv1d, apply_silu: bool = True) -> torch.Tensor:
        """Causal depthwise conv1d: (B, L, D) -> (B, L, D)."""
        y = conv(x.transpose(1, 2))          # (B, D, L + pad)
        y = y[..., :x.shape[1]].transpose(1, 2)  # truncate to causal
        if apply_silu:
            y = F.silu(y)
        return y

    # ---- delta rule recurrence --------------------------------------------

    @staticmethod
    def _delta_rule_recurrent(
        q: torch.Tensor,   # (B, H, L, K)
        k: torch.Tensor,   # (B, H, L, K)
        v: torch.Tensor,   # (B, H, L, V)
        beta: torch.Tensor, # (B, H, L)
    ) -> torch.Tensor:
        """Chunked parallel-scan delta rule from the Darts Reverso port."""
        B, H, L, K = q.shape
        V = v.shape[-1]
        device, dtype = q.device, q.dtype

        if K <= 8:
            chunk_size = 512
        elif K <= 16:
            chunk_size = 128
        else:
            chunk_size = 64

        eye = torch.eye(K, device=device, dtype=dtype)
        o = torch.empty(B, H, L, V, device=device, dtype=dtype)
        h = q.new_zeros(B, H, K, V)

        num_chunks = (L + chunk_size - 1) // chunk_size
        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, L)
            chunk_len = end - start

            q_c = q[:, :, start:end]
            k_c = k[:, :, start:end]
            v_c = v[:, :, start:end]
            beta_c = beta[:, :, start:end]

            beta_expanded = beta_c.unsqueeze(-1).unsqueeze(-1)
            transitions = eye - beta_expanded * (
                k_c.unsqueeze(-1) * k_c.unsqueeze(-2)
            )
            biases = beta_expanded * (
                k_c.unsqueeze(-1) * v_c.unsqueeze(-2)
            )

            scan_steps = int(math.ceil(math.log2(chunk_len)))
            for d in range(scan_steps):
                stride = 2**d
                if stride >= chunk_len:
                    break
                next_transitions = torch.matmul(
                    transitions[:, :, stride:],
                    transitions[:, :, : chunk_len - stride],
                )
                next_biases = (
                    torch.matmul(
                        transitions[:, :, stride:],
                        biases[:, :, : chunk_len - stride],
                    )
                    + biases[:, :, stride:]
                )
                transitions = torch.cat(
                    [transitions[:, :, :stride], next_transitions], dim=2
                )
                biases = torch.cat(
                    [biases[:, :, :stride], next_biases], dim=2
                )

            states = torch.matmul(
                transitions,
                h.unsqueeze(2).expand(-1, -1, chunk_len, -1, -1),
            ) + biases
            o[:, :, start:end] = torch.einsum(
                "bhlk,bhlkv->bhlv", q_c, states
            )
            h = transitions[:, :, -1] @ h + biases[:, :, -1]

        return o

    @staticmethod
    def _delta_rule_export(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Export-safe form of the same chunked parallel delta-rule scan."""
        _, _, sequence_length, key_dim = q.shape

        if key_dim <= 8:
            chunk_size = 512
        elif key_dim <= 16:
            chunk_size = 128
        else:
            chunk_size = 64

        eye = torch.eye(key_dim, device=q.device, dtype=q.dtype)
        state = q.new_zeros(
            q.shape[0],
            q.shape[1],
            key_dim,
            v.shape[-1],
        )
        chunks = []

        for start in range(0, sequence_length, chunk_size):
            end = min(start + chunk_size, sequence_length)
            chunk_length = end - start

            q_chunk = q[:, :, start:end]
            k_chunk = k[:, :, start:end]
            v_chunk = v[:, :, start:end]
            beta_chunk = beta[:, :, start:end]

            beta_expanded = beta_chunk.unsqueeze(-1).unsqueeze(-1)
            transitions = eye - beta_expanded * (
                k_chunk.unsqueeze(-1) * k_chunk.unsqueeze(-2)
            )
            biases = beta_expanded * (
                k_chunk.unsqueeze(-1) * v_chunk.unsqueeze(-2)
            )

            for depth in range(int(math.ceil(math.log2(chunk_length)))):
                stride = 2**depth
                next_transitions = torch.matmul(
                    transitions[:, :, stride:],
                    transitions[:, :, : chunk_length - stride],
                )
                next_biases = (
                    torch.matmul(
                        transitions[:, :, stride:],
                        biases[:, :, : chunk_length - stride],
                    )
                    + biases[:, :, stride:]
                )
                transitions = torch.cat(
                    [transitions[:, :, :stride], next_transitions],
                    dim=2,
                )
                biases = torch.cat(
                    [biases[:, :, :stride], next_biases],
                    dim=2,
                )

            states = torch.matmul(
                transitions,
                state.unsqueeze(2).expand(
                    -1,
                    -1,
                    chunk_length,
                    -1,
                    -1,
                ),
            ) + biases
            chunks.append(
                torch.einsum("bhlk,bhlkv->bhlv", q_chunk, states)
            )
            state = (
                transitions[:, :, -1] @ state
                + biases[:, :, -1]
            )

        return torch.cat(chunks, dim=2)

    # ---- forward ----------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs):
        B, L, _ = hidden_states.shape

        # projections + causal short conv (with fused SiLU for q/k/v)
        if self.use_short_conv:
            q = self._causal_conv1d(self.q_proj(hidden_states), self.q_conv1d, apply_silu=(self.qk_activation == 'silu'))
            k = self._causal_conv1d(self.k_proj(hidden_states), self.k_conv1d, apply_silu=(self.qk_activation == 'silu'))
            v = self._causal_conv1d(self.v_proj(hidden_states), self.v_conv1d, apply_silu=True)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            if self.qk_activation == 'silu':
                q = F.silu(q)
                k = F.silu(k)
            v = F.silu(v)

        # reshape to multi-head: (B, L, H, D)
        q = q.view(B, L, self.num_heads, self.head_k_dim)
        k = k.view(B, L, self.num_heads, self.head_k_dim)
        v = v.view(B, L, self.num_heads, self.head_v_dim)

        # L2 normalization per head (matches fla's l2norm: x / sqrt(||x||^2 + eps))
        if self.qk_norm == 'l2':
            q = q / (q.norm(2, dim=-1, keepdim=True).pow(2).add(1e-6)).sqrt()
            k = k / (k.norm(2, dim=-1, keepdim=True).pow(2).add(1e-6)).sqrt()

        # beta
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()   # (B, L, H)
        else:
            beta = q.new_ones(B, L, self.num_heads)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # -> (B, H, L, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        beta = beta.permute(0, 2, 1)          # (B, H, L)

        # scale q by 1/sqrt(head_k_dim) — matches fla's delta rule kernels
        q = q * (self.head_k_dim ** -0.5)

        # delta-rule linear attention
        if getattr(self, "browser_export", False):
            o = self._delta_rule_export(q, k, v, beta)
        else:
            o = self._delta_rule_recurrent(q, k, v, beta)

        # -> (B, L, H, V)  then RMSNorm per head
        o = o.permute(0, 2, 1, 3)
        o = self.o_norm(o)

        # merge heads and project
        o = o.reshape(B, L, self.value_dim)
        o = self.o_proj(o)
        return o, None, None


# ---------------------------------------------------------------------------
# AttentionBlock — uses TorchDeltaNet
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    def __init__(self, d_model, expand_v, state_weaving=False, is_intermediate=False):
        super().__init__()
        self.state_weaving = state_weaving
        self.is_intermediate = is_intermediate
        self.attention = TorchDeltaNet(
            mode='chunk',
            d_model=d_model,
            expand_k=1.0,
            expand_v=expand_v,
            num_heads=4,
            use_beta=True,
            use_gate=False,
            use_short_conv=True,
            conv_size=4,
            allow_neg_eigval=False,
            qk_activation='silu',
            qk_norm='l2',
            layer_idx=0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        residual = x_t
        if self.state_weaving and self.is_intermediate:
            x_t = torch.cat(
                [x_t[:, 0:1, :] + x_t[:, -1:, :], x_t[:, 1:, :]],
                dim=1,
            )
        attn_out = self.attention(hidden_states=x_t, attention_mask=None)
        if isinstance(attn_out, tuple):
            out = attn_out[0]
        else:
            out = attn_out
        out = self.norm(out)
        out = out + residual
        out = out.transpose(1, 2)
        return out


# ---------------------------------------------------------------------------
# Model — main architecture (no FlashFFTConv dependency)
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    Reverso (torch-native): conv-deltanet hybrid for time series forecasting.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.input_token_len = configs.input_token_len
        self.output_token_len = configs.output_token_len
        self.d_model = configs.d_model
        self.use_norm = configs.use_norm

        self.embedding = nn.Linear(1, self.d_model, bias=False)

        d_intermediate = configs.d_intermediate
        expand_v = getattr(configs, 'expand_v', 1.0)
        state_weaving = getattr(configs, 'state_weaving', False)
        gating_kernel_size = getattr(configs, 'gating_kernel_size', 3)
        module_list = [m.strip() for m in configs.main_module.split(',')]
        e_layers = len(module_list)

        layers = []
        for i, layer_type in enumerate(module_list):
            if layer_type == 'conv':
                layers.append(CNNBlock(
                    self.d_model, self.seq_len, gating_kernel_size,
                ))
            elif layer_type == 'attn':
                is_intermediate = (i > 0) and (i < e_layers - 1)
                layers.append(AttentionBlock(
                    self.d_model, expand_v, state_weaving, is_intermediate,
                ))
            else:
                raise ValueError(f'Invalid layer type: {layer_type}')
            layers.append(MLPBlock(self.d_model, self.d_model, d_intermediate))
        self.layers = nn.Sequential(*layers)

        output_bottleneck_dim = getattr(configs, 'output_bottleneck_dim', self.output_token_len)
        self.head = nn.Linear(self.input_token_len, output_bottleneck_dim, bias=configs.learn_bias)
        self.simple_q_proj = nn.Linear(self.d_model, self.d_model)
        self.key_proj = nn.Linear(self.d_model, self.d_model)
        self.value_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, 1)

        self.use_output_pe = getattr(configs, "use_output_pe", False)
        if self.use_output_pe:
            pe_max_len = self.seq_len + self.output_token_len
            self.output_position_embedding = PositionalEmbedding(
                self.d_model, max_len=pe_max_len
            )
            self.post_pe_q_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, x_mark=None, y_mark=None, **kwargs: Any):
        B, L, C = x.shape

        if self.use_norm:
            x_min = x.min(1, keepdim=True)[0].detach()
            x_max = x.max(1, keepdim=True)[0].detach()
            x_range = torch.clamp(x_max - x_min, min=1e-5).detach()
            x = (x - x_min) / x_range
            means = x_min
            stdev = x_range

        x = self.embedding(x).transpose(1, 2)

        dec_out = self.layers(x)

        temp_out = self.head(dec_out).permute(0, 2, 1)
        q = self.simple_q_proj(temp_out)

        dec_out_perm = dec_out.permute(0, 2, 1)
        if self.use_output_pe:
            full_hidden = torch.cat([dec_out_perm, q], dim=1)
            full_hidden = full_hidden + self.output_position_embedding(full_hidden)
            dec_out_pe = full_hidden[:, : dec_out_perm.shape[1], :]
            q = self.post_pe_q_proj(full_hidden[:, dec_out_perm.shape[1] :, :])
            k = self.key_proj(dec_out_pe)
            v = self.value_proj(dec_out_pe)
        else:
            k = self.key_proj(dec_out_perm)
            v = self.value_proj(dec_out_perm)

        if getattr(self, "browser_export", False):
            attention_weights = torch.softmax(
                torch.matmul(q, k.transpose(-1, -2))
                * (q.shape[-1] ** -0.5),
                dim=-1,
            )
            attn = torch.matmul(attention_weights, v)
        else:
            attn = F.scaled_dot_product_attention(q, k, v)
        dec_out = self.out_proj(attn)

        if self.use_norm:
            dec_out = dec_out * stdev + means

        return dec_out

    def forecast(self, x, x_mark=None, y_mark=None, **kwargs):
        return self.forward(x, x_mark, y_mark, **kwargs)
