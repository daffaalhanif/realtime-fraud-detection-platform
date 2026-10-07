"""Arsitektur Transformer kustom model tier 2: tokenizer per transaksi, encoder, dan dua kepala.

Tiap transaksi dalam sequence diringkas menjadi satu token, lalu encoder Transformer membaca
token-token itu dengan atensi yang mengabaikan padding. Output pada posisi terakhir (transaksi
yang dinilai) menjadi dasar skor fraud. Kepala pretraining menebak field yang disamarkan pada
semua posisi, dan hanya dipakai saat pretraining; graf yang diekspor ke ONNX cukup sampai
kepala klasifikasi.
"""

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fraud.features.sequence import FIRST_CATEGORY_CODE, SequenceSpec
from fraud.training.tier2.dataset import MAX_SEQ_LEN


@dataclass(frozen=True)
class Tier2Config:
    """Ukuran arsitektur model tier 2, dicatat ke MLflow bersama setiap run.

    Attributes:
        d_model: Dimensi token dan representasi encoder.
        n_layers: Jumlah blok encoder.
        n_heads: Jumlah kepala atensi, harus membagi habis `d_model`.
        d_ff: Dimensi lapisan tersembunyi feed-forward di tiap blok.
        dropout: Peluang dropout pada atensi, feed-forward, dan kepala klasifikasi.
        cat_dim: Dimensi embedding per kolom kategorikal.
        max_seq_len: Panjang sequence terpanjang yang didukung embedding posisi.
    """

    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.1
    cat_dim: int = 8
    max_seq_len: int = MAX_SEQ_LEN

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PretrainOutputs:
    """Tebakan kepala pretraining untuk setiap posisi.

    Attributes:
        categorical_logits: Satu tensor `(B, L, vocab kolom)` per kolom kategorikal.
        numeric: Tebakan nilai numerik ternormalisasi, `(B, L, kolom numerik)`.
        missing_logits: Logit indikator nilai kosong, `(B, L, kolom numerik)`.
    """

    categorical_logits: list[torch.Tensor]
    numeric: torch.Tensor
    missing_logits: torch.Tensor


def category_vocab_sizes(spec: SequenceSpec) -> list[int]:
    """Ukuran vocabulary tiap kolom kategorikal: kode khusus ditambah kategori aktif."""
    return [
        FIRST_CATEGORY_CODE + len(spec["category_codes"][name])
        for name in spec["categorical_columns"]
    ]


class TransactionTokenizer(nn.Module):
    """Meringkas field satu transaksi, jarak waktunya, dan posisinya menjadi satu token.

    Tiap kolom kategorikal punya embedding sendiri dengan satu kode tambahan di ujung
    vocabulary (`mask_codes`) untuk field yang disamarkan saat pretraining.
    """

    def __init__(self, spec: SequenceSpec, config: Tier2Config) -> None:
        super().__init__()
        vocab_sizes = category_vocab_sizes(spec)
        n_numeric = len(spec["numeric_columns"])
        self.embeddings = nn.ModuleList(
            nn.Embedding(size + 1, config.cat_dim) for size in vocab_sizes
        )
        self.register_buffer("mask_codes", torch.tensor(vocab_sizes), persistent=False)
        # Nilai, indikator kosong, dan indikator disamarkan untuk tiap kolom numerik.
        input_dim = len(vocab_sizes) * config.cat_dim + 3 * n_numeric
        self.project = nn.Sequential(
            nn.Linear(input_dim, config.d_model), nn.LayerNorm(config.d_model)
        )
        self.elapsed = nn.Linear(1, config.d_model)
        self.position = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(
        self,
        numeric: torch.Tensor,
        missing: torch.Tensor,
        categorical: torch.Tensor,
        elapsed: torch.Tensor,
        masked: torch.Tensor,
    ) -> torch.Tensor:
        embedded = [
            embedding(categorical[..., column]) for column, embedding in enumerate(self.embeddings)
        ]
        fields = torch.cat([*embedded, numeric, missing.float(), masked.float()], dim=-1)
        length = numeric.shape[1]
        # Posisi dihitung dari kanan: transaksi yang dinilai selalu posisi 0 apa pun panjangnya.
        positions = torch.arange(length - 1, -1, -1, device=numeric.device)
        return self.project(fields) + self.elapsed(elapsed.unsqueeze(-1)) + self.position(positions)


class EncoderBlock(nn.Module):
    """Blok Transformer pre-LN: atensi multi-kepala lalu feed-forward, masing-masing residual."""

    def __init__(self, config: Tier2Config) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError(
                f"d_model {config.d_model} tidak habis dibagi {config.n_heads} kepala."
            )
        self.n_heads = config.n_heads
        self.dropout = config.dropout
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.attention_out = nn.Linear(config.d_model, config.d_model)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: torch.Tensor, attend: torch.Tensor) -> torch.Tensor:
        batch, length, width = tokens.shape
        head_dim = width // self.n_heads
        query, key, value = (
            part.reshape(batch, length, self.n_heads, head_dim).transpose(1, 2)
            for part in self.qkv(self.attention_norm(tokens)).chunk(3, dim=-1)
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attend,
            dropout_p=self.dropout if self.training else 0.0,
        )
        tokens = tokens + self.attention_out(attended.transpose(1, 2).reshape(batch, length, width))
        return tokens + self.feed_forward(self.feed_forward_norm(tokens))


class Tier2Model(nn.Module):
    """Model tier 2: skor fraud transaksi terakhir dari sequence transaksi mentah entitasnya.

    `forward` adalah jalur inferensi yang diekspor ke ONNX. `encode` dan `pretrain_outputs`
    dipakai pretraining, yang juga membutuhkan saluran field disamarkan.
    """

    def __init__(self, spec: SequenceSpec, config: Tier2Config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = TransactionTokenizer(spec, config)
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        n_numeric = len(spec["numeric_columns"])
        self.pretrain_categorical = nn.ModuleList(
            nn.Linear(config.d_model, size) for size in category_vocab_sizes(spec)
        )
        self.pretrain_numeric = nn.Linear(config.d_model, n_numeric)
        self.pretrain_missing = nn.Linear(config.d_model, n_numeric)

    def encode(
        self,
        numeric: torch.Tensor,
        missing: torch.Tensor,
        categorical: torch.Tensor,
        elapsed: torch.Tensor,
        padding_mask: torch.Tensor,
        masked: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Representasi tiap posisi, `(B, L, d_model)`.

        Args:
            numeric: Nilai numerik ternormalisasi, `(B, L, kolom numerik)`.
            missing: Indikator nilai kosong, berbentuk sama dengan `numeric`.
            categorical: Kode kategori, `(B, L, kolom kategorikal)`; field disamarkan memakai
                `tokenizer.mask_codes`.
            elapsed: Jarak waktu ter-encode ke transaksi yang dinilai, `(B, L)`.
            padding_mask: True untuk posisi padding, `(B, L)`.
            masked: Indikator field numerik yang disamarkan; None berarti tidak ada.
        """
        if masked is None:
            masked = torch.zeros_like(numeric)
        tokens = self.tokenizer(numeric, missing, categorical, elapsed, masked)
        # True berarti boleh diperhatikan. Transaksi yang dinilai tidak pernah padding, jadi
        # tidak ada baris atensi yang seluruhnya tertutup.
        attend = ~padding_mask[:, None, None, :]
        for block in self.blocks:
            tokens = block(tokens, attend)
        return self.final_norm(tokens)

    def forward(
        self,
        numeric: torch.Tensor,
        missing: torch.Tensor,
        categorical: torch.Tensor,
        elapsed: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Logit fraud transaksi terakhir tiap sequence, `(B,)`."""
        hidden = self.encode(numeric, missing, categorical, elapsed, padding_mask)
        return self.classifier(hidden[:, -1]).squeeze(-1)

    def pretrain_outputs(self, hidden: torch.Tensor) -> PretrainOutputs:
        """Tebakan field yang disamarkan dari representasi hasil `encode`."""
        return PretrainOutputs(
            categorical_logits=[head(hidden) for head in self.pretrain_categorical],
            numeric=self.pretrain_numeric(hidden),
            missing_logits=self.pretrain_missing(hidden),
        )

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Parameter per kelompok untuk learning rate berbeda saat fine-tuning.

        Returns:
            `tokenizer`, `encoder` (blok dan normalisasi akhir), `classifier`, dan `pretrain_head`.
        """
        return {
            "tokenizer": list(self.tokenizer.parameters()),
            "encoder": [*self.blocks.parameters(), *self.final_norm.parameters()],
            "classifier": list(self.classifier.parameters()),
            "pretrain_head": [
                *self.pretrain_categorical.parameters(),
                *self.pretrain_numeric.parameters(),
                *self.pretrain_missing.parameters(),
            ],
        }
