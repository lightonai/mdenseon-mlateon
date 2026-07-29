"""
Contrastive + KL-div fine-tuning of a ColBERT model on the multilingual and code datasets
(query + 1 positive + stored hard negatives + cross-encoder teacher scores).
The script is adapted for late interaction long context training under limited GPU memory.

What to tweak
-------------
CLI:
  --model_name        Checkpoint to fine-tune (typically the pretraining output).
  --learning_rate / --num_train_epochs
  --batch_size        Global batch, split across devices; defines the pool of
                      in-batch negatives, which are gathered across devices, so
                      the real pool is batch_size * (1 + sampled negatives).
  --temperature       Temperature of the contrastive loss.
  --student_temperature / --teacher_temperature
                      Softmax temperatures of the KL-div loss applied to
                      student and teacher relevance scores. Sharp temperatures
                      distill the teacher ranking rather than its margins.
  --contrastive_weight / --kldiv_weight
  --query_length / --document_length
                      Query and document token budgets.
  --mini_batch_size   GradCache chunk size to reduce GPU memory usage.
  --alpha             Multinomial sampling exponent: each batch is drawn from a
                      dataset with probability proportional to size ** alpha
                      (1.0 = proportional to size, 0.5 = smoothed, 0 = uniform).

Data build (load_train_datasets — baked into the on-disk cache, delete
{cache_dir}/{split} after changing any of them):
  code_splits / multilingual_splits  Which subsets to train on.
  nv_threshold=0.95   A document counts as a negative only if its retrieval
                      score is < 0.95 * positive score (false-negative filter;
                      lower = stricter).
  num_negatives=10    Hard negatives stored per query, with their teacher
                      rerank scores aligned to [positive, negative_0, ...].

In main():
  num_negatives=7 (collator)  Negatives sampled per step from the stored 10.
  do_query_expansion=False    Query MASK expansion; True pads every query out to
                              query_length, which is ruinous at 8192.
  skiplist_words=[]           Document tokens excluded from the MaxSim.
  eval chunk sizes            MultilingualNanoBEIREvaluator's corpus_chunk_size and
                              score token chunk sizes; eval-only memory knobs.
  loss GPU memory knobs       CachedContrastiveKLDiv's chunk_token_budget,
                              score_bytes_budget and score_anchor_token_budget cap
                              the tokens of an encoder chunk and the bytes of a
                              score tile; lower them if a chunk or tile OOMs.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import logging
import os
import random
from datetime import timedelta
from functools import partial
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed

set_seed(42)

from datasets import Dataset, DatasetDict, load_dataset
from pylate import evaluation, models
from pylate.losses.cached_contrastive import RandContext
from pylate.losses.contrastive import extract_skiplist_mask
from pylate.utils import all_gather, all_gather_with_gradients, get_rank, get_world_size
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import NanoBEIREvaluator as NanoBEIREvaluatorST
from sentence_transformers.model_card import SentenceTransformerModelCardData
from sentence_transformers.sampler import MultiDatasetDefaultBatchSampler
from tqdm import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)

# Disable dataset metrics
SentenceTransformerModelCardData.compute_dataset_metrics = lambda self, dataset, dataset_info, loss: {}


class CachedContrastiveKLDiv(torch.nn.Module):
    """Gradient-cached contrastive and KL-divergence distillation loss for ColBERT model.

    A single chunked encoder pass feeds two terms. The contrastive term is a cross-entropy over
    the MeanMaxSim scores of every in-batch document of every rank, so the other queries'
    positives and all of the negatives act as negatives. The KL-div term is, per query, the
    KL divergence between the student distribution over its own (positive, negative_0, ...,
    negative_k) and the teacher cross-encoder scores of those same documents, each side
    softmaxed at its own temperature. The loss is
    ``contrastive_weight * contrastive + kldiv_weight * kldiv``.

    The GradCache implementation is optimized for batches mixing short and long texts
    and works faster under limited GPU memory.

    Parameters
    ----------
    model
        ColBERT model.
    contrastive_temperature
        Temperature applied to the student scores of the contrastive term.
    student_temperature
        Softmax temperature applied to the student scores of the KL-div term.
    teacher_temperature
        Softmax temperature applied to the teacher scores of the KL-div term.
    contrastive_weight
        Weight of the contrastive term.
    kldiv_weight
        Weight of the KL-div term, 0 trains on the contrastive term alone.
    mini_batch_size
        GradCache chunk size, a pure memory knob that does not affect the loss.
    defer_grad_sync
        Whether to accumulate the re-forward chunk gradients locally and allreduce once.
    score_bytes_budget
        Cap on the bytes of a single MaxSim score tile.
    chunk_token_budget
        Maximum number of tokens allowed in one encoder chunk, rows x the chunk's max length.
    score_anchor_token_budget
        Cap on the padded query tokens (rows x query max length) of one scoring chunk.
    rand_context
        Whether to capture and replay the RNG state of each chunk, needed only to replay
        dropout: "auto" does it when the model has active dropout, "always" and "never" force it.

    Requirements
    ------------
    1. Columns ordered (query, positive, negative_0, ..., negative_k).
    2. When kldiv_weight is not 0, labels of shape (batch_size, 1 + k) holding the teacher
       scores of those same documents, in the same order.

    Examples
    --------
    >>> from pylate import models

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> loss = CachedContrastiveKLDiv(model=model, mini_batch_size=1)

    >>> query = model.tokenize(["fruits are healthy."], is_query=True, pad=False)

    >>> positive = model.tokenize(["fruits are good for health."], is_query=False, pad=False)

    >>> negative = model.tokenize(["fruits are bad for health."], is_query=False, pad=False)

    >>> labels = torch.tensor([[0.7, 0.3]], dtype=torch.float32)

    >>> output = loss(sentence_features=[query, positive, negative], labels=labels)

    >>> assert isinstance(output.item(), float)
    """

    class _FusedPlan:
        """Length-sorted chunking plan over the concatenation of all columns."""

        def __init__(
            self,
            sentence_features: list[dict[str, torch.Tensor]],
            row_cap: int,
            token_budget: int | None,
        ) -> None:
            device = sentence_features[0]["attention_mask"].device
            self.col_lengths = [sf["attention_mask"].sum(dim=1) for sf in sentence_features]
            col_sizes = [int(l.size(0)) for l in self.col_lengths]
            all_lens = torch.cat(self.col_lengths)
            col_ids = torch.cat(
                [
                    torch.full((n,), c, dtype=torch.long, device=device)
                    for c, n in enumerate(col_sizes)
                ]
            )
            row_ids = torch.cat([torch.arange(n, device=device) for n in col_sizes])
            order = torch.argsort(all_lens, descending=True, stable=True)
            # Rows are grouped by column within each chunk so the encoder input is one index_select + cat per column
            self.chunks: list[tuple[int, int, int]] = []
            self.chunk_cols: list[torch.Tensor] = []  # column id per row, chunk order
            self.chunk_rows: list[torch.Tensor] = []  # original row id per row
            sorted_cols = col_ids.index_select(0, order)
            sorted_rows = row_ids.index_select(0, order)
            sorted_lens = [int(x) for x in all_lens.index_select(0, order).tolist()]
            for begin, end, chunk_len in CachedContrastiveKLDiv._greedy_chunks(
                sorted_lens, row_cap, token_budget, waste_cap=1.5
            ):
                cols = sorted_cols[begin:end]
                rows = sorted_rows[begin:end]
                grouped = torch.argsort(cols, stable=True)
                self.chunks.append((begin, end, chunk_len))
                self.chunk_cols.append(cols.index_select(0, grouped))
                self.chunk_rows.append(rows.index_select(0, grouped))
            self.num_columns = len(sentence_features)
            self.col_max = [
                int(l.max().item()) if l.numel() else 0 for l in self.col_lengths
            ]

    @staticmethod
    def _all_gather_padded(tensor: torch.Tensor, with_gradients: bool):
        """Cross-rank all_gather for tensors whose dim-1 (sequence) differs by rank."""
        if get_world_size() == 1:
            return [tensor]
        local_len = torch.tensor([tensor.size(1)], device=tensor.device)
        max_len = int(torch.cat(all_gather(local_len)).max().item())
        if tensor.size(1) < max_len:
            pad = [0] * (2 * (tensor.dim() - 1))
            pad[-1] = max_len - tensor.size(1)
            tensor = F.pad(tensor, pad)
        if with_gradients:
            return all_gather_with_gradients(tensor)
        return all_gather(tensor)

    @staticmethod
    def _greedy_chunks(
        sorted_lengths: list[int],
        row_cap: int,
        token_budget: int | None,
        waste_cap: float | None = None,
        min_tokens: int = 8192,
    ) -> list[tuple[int, int, int]]:
        """(begin, end, chunk_max_len) over a descending-length row order.

        A chunk grows while within row_cap and token_budget; with waste_cap set it
        additionally stops growing once padded/real exceeds the cap - unless the
        chunk is still below min_tokens padded (avoids tiny launch-bound chunks).
        """
        chunks = []
        n = len(sorted_lengths)
        begin = 0
        while begin < n:
            chunk_len = max(1, sorted_lengths[begin])
            real = chunk_len
            end = begin + 1
            while end < n and end - begin < row_cap:
                rows = end - begin + 1
                padded = rows * chunk_len
                if token_budget is not None and padded > token_budget:
                    break
                if (
                    waste_cap is not None
                    and padded >= min_tokens
                    and padded > waste_cap * (real + sorted_lengths[end])
                ):
                    break
                real += sorted_lengths[end]
                end += 1
            chunks.append((begin, end, chunk_len))
            begin = end
        return chunks

    def __init__(
        self,
        model,
        contrastive_temperature: float = 0.02,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        contrastive_weight: float = 1.0,
        kldiv_weight: float = 1.0,
        mini_batch_size: int = 8,
        defer_grad_sync: bool = True,
        score_bytes_budget: int = 512 << 20,
        chunk_token_budget: int | None = 65536,
        score_anchor_token_budget: int = 65536,
        rand_context: str = "auto",  # "auto" | "always" | "never"
    ) -> None:
        super().__init__()
        self.model = model
        self.contrastive_temperature = contrastive_temperature
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.contrastive_weight = contrastive_weight
        self.kldiv_weight = kldiv_weight
        self.mini_batch_size = mini_batch_size
        self.defer_grad_sync = defer_grad_sync
        self.score_bytes_budget = score_bytes_budget
        self.chunk_token_budget = chunk_token_budget
        self.score_anchor_token_budget = score_anchor_token_budget
        self.rand_context = rand_context

        self.cache: list[list[torch.Tensor]] | None = None
        self.random_states: list[list[RandContext | None]] | None = None
        self._needs_rand_context: bool | None = None

    # utils
    def _use_rand_context(self) -> bool:
        if self.rand_context == "always":
            return True
        if self.rand_context == "never":
            return False
        if self._needs_rand_context is None:
            module = self.model.module if hasattr(self.model, "module") else self.model
            has_dropout = any(
                isinstance(m, torch.nn.Dropout) and m.p > 0 for m in module.modules()
            )
            cfg = getattr(getattr(module, "_first_module", lambda: None)(), "auto_model", None)
            cfg = getattr(cfg, "config", None)
            if cfg is not None:
                for k, v in cfg.to_dict().items():
                    if k.endswith("dropout") and isinstance(v, (int, float)) and v > 0:
                        has_dropout = True
            self._needs_rand_context = has_dropout
        return self._needs_rand_context

    def _fused_plan(self, sentence_features: list[dict[str, torch.Tensor]]) -> _FusedPlan:
        # the token budget is the real bound; 4x mini_batch_size rows just keeps kernel sizes sane on short splits
        return self._FusedPlan(
            sentence_features,
            row_cap=max(1, self.mini_batch_size) * 4,
            token_budget=self.chunk_token_budget,
        )

    @staticmethod
    def _chunk_features(
        sentence_features: list[dict[str, torch.Tensor]],
        cols: torch.Tensor,
        rows: torch.Tensor,
        chunk_len: int,
    ) -> dict[str, torch.Tensor]:
        """Build one fused encoder chunk (rows grouped by column)."""
        keys = set(sentence_features[0].keys())
        for sf in sentence_features[1:]:
            keys &= set(sf.keys())
        parts: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
        for c in torch.unique_consecutive(cols).tolist():
            sel = rows[cols == c]
            sf = sentence_features[c]
            for k in keys:
                v = sf[k]
                if not (isinstance(v, torch.Tensor) and v.dim() >= 2):
                    continue
                piece = v.index_select(0, sel)
                if piece.size(1) >= chunk_len:
                    piece = piece[:, :chunk_len]
                else:
                    piece = F.pad(piece, (0, chunk_len - piece.size(1)))
                parts[k].append(piece)
        return {k: torch.cat(v, dim=0) for k, v in parts.items() if v}

    # encoder chunked pass
    def embed_minibatch_iter(
        self,
        sentence_features: list[dict[str, torch.Tensor]],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext | None] | None = None,
    ) -> Iterator[tuple[torch.Tensor, RandContext | None]]:
        plan = self._fused_plan(sentence_features)
        use_rand = self._use_rand_context()
        for i, (begin, end, chunk_len) in enumerate(plan.chunks):
            chunk = self._chunk_features(
                sentence_features, plan.chunk_cols[i], plan.chunk_rows[i], chunk_len
            )
            random_state = None if random_states is None else random_states[i]
            grad_context = contextlib.nullcontext if with_grad else torch.no_grad
            random_state_context = (
                contextlib.nullcontext() if random_state is None else random_state
            )
            with random_state_context:
                with grad_context():
                    new_state = (
                        RandContext(*chunk.values())
                        if copy_random_state and use_rand
                        else None
                    )
                    embeddings = F.normalize(
                        self.model(chunk)["token_embeddings"], p=2, dim=-1
                    )
            yield embeddings, new_state

    def _assemble_columns(
        self, reps: list[torch.Tensor], plan: _FusedPlan
    ) -> list[torch.Tensor]:
        """Rebuild per-column (n_rows, col_max_len, h) tensors in original row order."""
        pieces: list[list[torch.Tensor]] = [[] for _ in range(plan.num_columns)]
        piece_rows: list[list[torch.Tensor]] = [[] for _ in range(plan.num_columns)]
        for chunk_emb, cols, rows in zip(reps, plan.chunk_cols, plan.chunk_rows):
            for c in torch.unique_consecutive(cols).tolist():
                sel = (cols == c).nonzero(as_tuple=True)[0]
                target = plan.col_max[c]
                piece = chunk_emb.index_select(0, sel)
                if piece.size(1) < target:
                    piece = F.pad(piece, (0, 0, 0, target - piece.size(1)))
                else:
                    piece = piece[:, :target]
                pieces[c].append(piece)
                piece_rows[c].append(rows.index_select(0, sel))
        out = []
        for c in range(plan.num_columns):
            emb = torch.cat(pieces[c], dim=0)
            rows_c = torch.cat(piece_rows[c])
            inverse = torch.empty_like(rows_c)
            inverse[rows_c] = torch.arange(rows_c.size(0), device=rows_c.device)
            out.append(emb.index_select(0, inverse))
        return out

    # loss
    def calculate_loss_and_cache_gradients(
        self,
        reps: list[list[torch.Tensor]],
        masks: list[torch.Tensor],
        labels: torch.Tensor,
        plan: _FusedPlan,
    ) -> torch.Tensor:
        loss = self.calculate_loss(reps, masks, labels, plan, with_backward=True)
        loss = loss.detach().requires_grad_()
        self.cache = [[r.grad for r in rs] for rs in reps]
        return loss

    def calculate_loss(
        self,
        reps: list[list[torch.Tensor]],
        masks: list[torch.Tensor],
        labels: torch.Tensor,
        plan: _FusedPlan,
        with_backward: bool = False,
    ) -> torch.Tensor:
        device = reps[0][0].device
        do_query_expansion = (
            self.model.do_query_expansion
            if hasattr(self.model, "do_query_expansion")
            else self.model.module.do_query_expansion
        )

        embeddings = self._assemble_columns(reps[0], plan)
        bs_local = embeddings[0].size(0)
        world = get_world_size()
        self_offset = get_rank() * bs_local

        shared_tensors: list[torch.Tensor] = []
        shared_leaves: list[torch.Tensor] = []

        def leaf_boundary(t: torch.Tensor) -> torch.Tensor:
            if not with_backward:
                return t
            shared_tensors.append(t)
            leaf = t.detach().requires_grad_()
            shared_leaves.append(leaf)
            return leaf

        # anchors: fold query mask, detach into a leaf
        anchor_mask = (
            masks[0][:, : plan.col_max[0]] if not do_query_expansion else None
        )
        anchor_emb = embeddings[0]
        if anchor_mask is not None:
            anchor_emb = anchor_emb * anchor_mask.unsqueeze(-1).to(anchor_emb.dtype)
            q_denom_full = anchor_mask.sum(dim=-1).clamp(min=1).unsqueeze(1).to(anchor_emb.dtype)
            anchor_lens = anchor_mask.sum(dim=1)
        else:
            q_denom_full = torch.full(
                (bs_local, 1), anchor_emb.size(1), device=device, dtype=anchor_emb.dtype
            )
            anchor_lens = torch.full((bs_local,), anchor_emb.size(1), device=device)
        anchor_emb = leaf_boundary(anchor_emb)

        # documents: fold masks, gather, length-sort, detach into leaves; docs below the column max clamp at >= 0
        doc_groups = []  # (leaf, lens_sorted, inv_perm, clamp_sorted)
        for emb, mask, col_max in zip(embeddings[1:], masks[1:], plan.col_max[1:]):
            folded = emb * mask[:, :col_max].unsqueeze(-1).to(emb.dtype)
            gathered = torch.cat(self._all_gather_padded(folded, with_gradients=True))
            lens = torch.cat(all_gather(mask[:, :col_max].sum(dim=1)))
            t_max = gathered.size(1)

            perm = torch.argsort(lens, descending=True, stable=True)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(perm.size(0), device=device)
            emb_sorted = gathered.index_select(0, perm)
            lens_sorted = [int(x) for x in lens.index_select(0, perm).tolist()]
            clamp_sorted = torch.tensor([l < t_max for l in lens_sorted], device=device)

            doc_groups.append(
                (leaf_boundary(emb_sorted), lens_sorted, inv_perm, clamp_sorted)
            )

        use_kl = labels is not None and self.kldiv_weight > 0
        if use_kl:
            teacher_scores = labels.to(device=device, dtype=anchor_emb.dtype)
            teacher_log_probs = F.log_softmax(
                teacher_scores / self.teacher_temperature, dim=-1
            )

        contrastive_total = torch.zeros((), device=device)
        kl_total = torch.zeros((), device=device)

        # anchor scoring chunks: length-sorted and token-budgeted, so long-query splits get short tiles
        a_order = torch.argsort(anchor_lens, descending=True, stable=True)
        a_lens_sorted = [int(x) for x in anchor_lens.index_select(0, a_order).tolist()]
        for begin, end, s_len in self._greedy_chunks(
            a_lens_sorted,
            row_cap=self.mini_batch_size,
            token_budget=self.score_anchor_token_budget,
        ):
            orig_rows = a_order[begin:end]
            a_emb = anchor_emb.index_select(0, orig_rows)[:, :s_len]
            a_size = a_emb.size(0)
            q_denom = q_denom_full.index_select(0, orig_rows)

            per_group_scores = []
            for leaf, lens_sorted, inv_perm, clamp_sorted in doc_groups:
                # doc pieces sized from the TRUE tile dims so the byte budget holds, with a floor of 1 doc
                b_total = leaf.size(0)
                score_pieces = []
                piece_bounds = []
                start = 0
                while start < b_total:
                    t_piece = max(1, lens_sorted[start])
                    budget_docs = self.score_bytes_budget // max(
                        1, a_size * s_len * t_piece * leaf.element_size()
                    )
                    end_ = min(b_total, start + max(1, budget_docs))
                    piece_bounds.append((start, end_, t_piece))
                    start = end_
                for g_start, g_end, t_piece in piece_bounds:
                    sim = torch.einsum(
                        "ash,bth->abst", a_emb, leaf[g_start:g_end, :t_piece]
                    )
                    piece_max = sim.max(dim=-1).values  # (a, p, s)
                    clamp = clamp_sorted[g_start:g_end]
                    if bool(clamp.any()):
                        piece_max = torch.where(
                            clamp.view(1, -1, 1), piece_max.clamp_min(0), piece_max
                        )
                    score_pieces.append(piece_max.sum(dim=-1))
                scores_sorted = torch.cat(score_pieces, dim=1)  # (a, B_total)
                # divide by the query length -> MeanMaxSim, independent of query length
                per_group_scores.append(
                    scores_sorted.index_select(1, inv_perm) / q_denom
                )

            chunk_scores = torch.cat(per_group_scores, dim=1)
            targets = self_offset + orig_rows
            ce_sum = F.cross_entropy(
                chunk_scores / self.contrastive_temperature, targets, reduction="sum"
            ) * world
            chunk_loss = self.contrastive_weight * ce_sum

            kl_sum = None
            if use_kl:
                student_scores = torch.stack(
                    [g.gather(1, targets.unsqueeze(1)).squeeze(1) for g in per_group_scores],
                    dim=1,
                )
                student_log_probs = F.log_softmax(
                    student_scores / self.student_temperature, dim=-1
                )
                kl_sum = F.kl_div(
                    student_log_probs,
                    teacher_log_probs.index_select(0, orig_rows),
                    reduction="sum",
                    log_target=True,
                )
                chunk_loss = chunk_loss + self.kldiv_weight * kl_sum

            if with_backward:
                (chunk_loss / bs_local).backward()
                ce_sum = ce_sum.detach()
                kl_sum = kl_sum.detach() if kl_sum is not None else None
            contrastive_total = contrastive_total + ce_sum
            if kl_sum is not None:
                kl_total = kl_total + kl_sum

        if with_backward and shared_tensors:
            # one traversal of each shared path (assembly, mask folds, gather collectives) with the accumulated grads
            torch.autograd.backward(
                tensors=shared_tensors,
                grad_tensors=[leaf.grad for leaf in shared_leaves],
            )

        total = self.contrastive_weight * contrastive_total / bs_local
        if use_kl:
            total = total + self.kldiv_weight * kl_total / bs_local
        return total

    # forward
    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sentence_features = list(sentence_features)

        skiplist = (
            self.model.skiplist
            if hasattr(self.model, "skiplist")
            else self.model.module.skiplist
        )
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )
        plan = self._fused_plan(sentence_features)

        reps_mbs: list[torch.Tensor] = []
        random_state_mbs: list[RandContext | None] = []
        for reps_mb, random_state in self.embed_minibatch_iter(
            sentence_features=sentence_features,
            with_grad=False,
            copy_random_state=True,
        ):
            reps_mbs.append(reps_mb.detach().requires_grad_())
            random_state_mbs.append(random_state)
        reps = [reps_mbs]
        self.random_states = [random_state_mbs]

        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps, masks, labels, plan)
            loss.register_hook(
                lambda grad_output: self._backward_hook(grad_output, sentence_features)
            )
        else:
            loss = self.calculate_loss(reps, masks, labels, plan)
        return loss

    # backward hook
    def _backward_hook(
        self, grad_output: torch.Tensor, sentence_features: list[dict[str, torch.Tensor]]
    ) -> None:
        assert self.cache is not None and self.random_states is not None
        grads = self.cache[0]
        total_chunks = len(grads)
        can_no_sync = self.defer_grad_sync and hasattr(self.model, "no_sync")
        with torch.enable_grad():
            chunk_iter = self.embed_minibatch_iter(
                sentence_features=sentence_features,
                with_grad=True,
                copy_random_state=False,
                random_states=self.random_states[0],
            )
            for i, grad_mb in enumerate(grads):
                # DDP syncs on the last chunk only, the others accumulate locally (allreduce is linear)
                sync_ctx = (
                    self.model.no_sync()
                    if can_no_sync and i + 1 < total_chunks
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    reps_mb, _ = next(chunk_iter)
                    surrogate = (
                        torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                    )
                    surrogate.backward()


class MultilingualNanoBEIREvaluator(NanoBEIREvaluatorST):
    """Evaluate PyLate models on the Multilingual NanoBEIR collection, per language and aggregated."""

    DATASETS = {
        "climatefever": "ClimateFEVER", "dbpedia": "DBPedia", "fever": "FEVER",
        "fiqa2018": "FiQA2018", "hotpotqa": "HotpotQA", "msmarco": "MSMARCO",
        "nfcorpus": "NFCorpus", "nq": "NQ", "quoraretrieval": "QuoraRetrieval",
        "scidocs": "SCIDOCS", "arguana": "ArguAna", "scifact": "SciFact",
        "touche2020": "Touche2020",
    }
    SUPPORTED_LANGUAGES = ["ar", "de", "en", "es", "fr", "it", "no", "pt", "sv"]

    @staticmethod
    def _memory_efficient_colbert_scores(
        queries_embeddings,
        documents_embeddings,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
        query_token_chunk_size: int = 64,
        document_token_chunk_size: int = 300,
    ) -> torch.Tensor:
        """MaxSim scoring in query-token/document-token blocks.

        PyLate's default eval scorer materializes the full [Q, D, QTok, DTok] tensor,
        which is explosive for 8192-token documents; this computes the same reduction
        block by block.
        """
        queries_embeddings = torch.as_tensor(queries_embeddings)
        like = {"device": queries_embeddings.device, "dtype": queries_embeddings.dtype}
        documents_embeddings = torch.as_tensor(documents_embeddings).to(**like)
        queries_mask = None if queries_mask is None else torch.as_tensor(queries_mask).to(**like)
        documents_mask = None if documents_mask is None else torch.as_tensor(documents_mask).to(**like)

        query_length = queries_embeddings.size(1)
        document_length = documents_embeddings.size(1)
        scores = torch.zeros(
            (queries_embeddings.size(0), documents_embeddings.size(0)), **like
        )

        with torch.no_grad():
            for q_start in range(0, query_length, query_token_chunk_size):
                q_end = min(q_start + query_token_chunk_size, query_length)
                query_chunk = queries_embeddings[:, q_start:q_end]
                query_chunk_mask = (
                    queries_mask[:, q_start:q_end] if queries_mask is not None else None
                )
                max_scores = None

                for d_start in range(0, document_length, document_token_chunk_size):
                    d_end = min(d_start + document_token_chunk_size, document_length)
                    chunk_scores = torch.einsum(
                        "ash,bth->abst", query_chunk, documents_embeddings[:, d_start:d_end]
                    )
                    if query_chunk_mask is not None:
                        chunk_scores = chunk_scores * query_chunk_mask.unsqueeze(1).unsqueeze(3)
                    if documents_mask is not None:
                        chunk_scores = chunk_scores * documents_mask[:, d_start:d_end].unsqueeze(0).unsqueeze(2)
                    chunk_max_scores = chunk_scores.max(dim=-1).values
                    max_scores = (
                        chunk_max_scores
                        if max_scores is None
                        else torch.maximum(max_scores, chunk_max_scores)
                    )

                scores.add_(max_scores.sum(dim=-1))

        return scores

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        languages: list[str] | None = None,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        corpus_chunk_size: int = 128,
        score_query_token_chunk_size: int = 64,
        score_document_token_chunk_size: int = 300,
        write_csv: bool = True,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        query_prompts: str | dict[str, str] | None = None,
        corpus_prompts: str | dict[str, str] | None = None,
        dataset_path: str | None = None,
    ):
        self.dataset_names = dataset_names or list(self.DATASETS)
        self.languages = languages or self.SUPPORTED_LANGUAGES
        self.aggregate_fn = aggregate_fn
        self.aggregate_key = aggregate_key
        self.write_csv = write_csv
        self.query_prompts = query_prompts
        self.corpus_prompts = corpus_prompts
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.dataset_path = dataset_path or "lightonai/nanobeir-multilingual"

        self.name = f"MultilingualNanoBEIR_{aggregate_key}"

        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self._validate_dataset_names()
        self._validate_languages()
        self._validate_prompts()

        ir_evaluator_kwargs = {
            "mrr_at_k": mrr_at_k,
            "ndcg_at_k": ndcg_at_k,
            "accuracy_at_k": accuracy_at_k,
            "precision_recall_at_k": precision_recall_at_k,
            "map_at_k": map_at_k,
            "show_progress_bar": show_progress_bar,
            "batch_size": batch_size,
            "corpus_chunk_size": corpus_chunk_size,
            "score_functions": {
                "MaxSim": partial(
                    self._memory_efficient_colbert_scores,
                    query_token_chunk_size=score_query_token_chunk_size,
                    document_token_chunk_size=score_document_token_chunk_size,
                )
            },
            "write_csv": write_csv,
        }

        self.evaluators = []
        self.evaluator_index = {}
        for lang in self.languages:
            for name in self.dataset_names:
                try:
                    evaluator = self._load_dataset(name, lang, **ir_evaluator_kwargs)
                    self.evaluators.append(evaluator)
                    self.evaluator_index[(name, lang)] = evaluator
                except Exception as e:
                    logger.warning(f"Failed to load {name}-{lang}: {e}")

        if not self.evaluators:
            raise ValueError(
                "No evaluators created. Please check your dataset_names and languages."
            )

        self.csv_file = f"MultilingualNanoBEIR_PyLate_{aggregate_key}_results.csv"
        self.csv_headers = self._build_csv_headers()

    def _metric_keys(self) -> list[str]:
        return [
            *[f"MaxSim_accuracy@{k}" for k in self.accuracy_at_k],
            *itertools.chain.from_iterable(
                (f"MaxSim_precision@{k}", f"MaxSim_recall@{k}")
                for k in self.precision_recall_at_k
            ),
            *[f"MaxSim_mrr@{k}" for k in self.mrr_at_k],
            *[f"MaxSim_ndcg@{k}" for k in self.ndcg_at_k],
            *[f"MaxSim_map@{k}" for k in self.map_at_k],
        ]

    def _build_csv_headers(self) -> list[str]:
        return ["epoch", "steps", *self._metric_keys()]

    def __call__(
        self, model, output_path: str | None = None, epoch: int = -1, steps: int = -1,
        *args, **kwargs,
    ) -> dict[str, float]:
        per_metric_results = {}
        per_dataset_results = {}
        per_language_results = {lang: {} for lang in self.languages}

        out_txt = ""
        if epoch != -1:
            out_txt = (
                f" after epoch {epoch}" if steps == -1
                else f" in epoch {epoch} after {steps} steps"
            )

        logger.info(
            f"Multilingual NanoBEIR PyLate Evaluation on {len(self.evaluators)} dataset-language pairs{out_txt}"
        )

        for evaluator in tqdm(
            self.evaluators, desc="Evaluating datasets", disable=not self.show_progress_bar,
        ):
            logger.info(f"Evaluating {evaluator.name}")
            results = evaluator(model, output_path, epoch, steps)
            lang = evaluator.name.split("-")[-1]  # names are "NanoDataset-lang"

            for full_key, metric_value in results.items():
                parts = full_key.split("_")
                metric = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                per_dataset_results[full_key] = metric_value
                per_metric_results.setdefault(metric, []).append(metric_value)
                if lang in per_language_results:
                    per_language_results[lang].setdefault(metric, []).append(metric_value)

        agg_results = {
            f"{self.name}_{metric}": self.aggregate_fn(values)
            for metric, values in per_metric_results.items()
        }
        per_lang_agg_results = {
            f"{self.name}_{lang}_{metric}": self.aggregate_fn(values)
            for lang in self.languages
            for metric, values in per_language_results[lang].items()
            if values
        }

        if output_path is not None and self.write_csv:
            self._write_csv(output_path, epoch, steps, agg_results)
            self._write_per_language_csv(output_path, epoch, steps, per_lang_agg_results)

        self._log_results(agg_results)
        self._log_per_language_results(per_lang_agg_results)

        if not getattr(self, "primary_metric", None):
            self.primary_metric = f"{self.name}_MaxSim_ndcg@{max(self.ndcg_at_k)}"

        per_dataset_results.update(agg_results)
        per_dataset_results.update(per_lang_agg_results)
        return per_dataset_results

    @staticmethod
    def _append_csv(csv_path: str, headers: list[str], rows: list[list]):
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        if not os.path.isfile(csv_path):
            with open(csv_path, mode="w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
        with open(csv_path, mode="a", encoding="utf-8") as f:
            for row in rows:
                f.write(",".join(map(str, row)) + "\n")

    def _write_csv(self, output_path, epoch, steps, agg_results):
        row = [epoch, steps] + [
            agg_results.get(f"{self.name}_{key}", "") for key in self._metric_keys()
        ]
        self._append_csv(
            os.path.join(output_path, self.csv_file), self.csv_headers, [row]
        )

    def _write_per_language_csv(self, output_path, epoch, steps, per_lang_results):
        rows = [
            [epoch, steps, lang]
            + [
                per_lang_results.get(f"{self.name}_{lang}_{key}", "")
                for key in self._metric_keys()
            ]
            for lang in self.languages
        ]
        self._append_csv(
            os.path.join(
                output_path,
                f"MultilingualNanoBEIR_PyLate_{self.aggregate_key}_per_language.csv",
            ),
            ["epoch", "steps", "language", *self._metric_keys()],
            rows,
        )

    def _log_metrics(self, results: dict, prefix: str, indent: str):
        for key in self._metric_keys():
            value = results.get(f"{prefix}{key}", 0)
            scaled = "accuracy" in key or "precision" in key or "recall" in key
            logger.info(
                f"{indent}{key}: {value * 100:.2f}%" if scaled else f"{indent}{key}: {value:.4f}"
            )

    def _log_results(self, agg_results):
        logger.info(f"\nAggregated Results ({self.aggregate_key}):")
        self._log_metrics(agg_results, f"{self.name}_", "  ")

    def _log_per_language_results(self, per_lang_results):
        logger.info(f"\nPer-Language Results ({self.aggregate_key}):")
        for lang in self.languages:
            logger.info(f"\n  Language: {lang}")
            self._log_metrics(per_lang_results, f"{self.name}_{lang}_", "    ")

    def _get_human_readable_name(self, dataset_name, language):
        return f"Nano{self.DATASETS[dataset_name.lower()]}-{language}"

    def _load_dataset(self, dataset_name, language, **ir_evaluator_kwargs):
        base_name = f"Nano{self.DATASETS[dataset_name.lower()]}"
        try:
            corpus = load_dataset(self.dataset_path, f"{base_name}_{language}", split="corpus")
            queries = load_dataset(self.dataset_path, f"{base_name}_{language}", split="queries")
            qrels = load_dataset(self.dataset_path, base_name, split="qrels")
        except Exception as e:
            raise ValueError(f"Failed to load dataset {base_name} for language {language}: {e}")

        # Length-sorting the corpus cuts the padding wasted inside each scoring chunk and keeps the top-k metrics
        corpus_items = sorted(
            ((s["_id"], s["text"]) for s in corpus if s.get("text")),
            key=lambda item: len(item[1]),
        )
        queries_dict = {s["_id"]: s["text"] for s in queries if s.get("text")}
        qrels_dict = {}
        for sample in qrels:
            qrels_dict.setdefault(sample["query-id"], set()).add(sample["corpus-id"])

        if self.query_prompts is not None:
            ir_evaluator_kwargs["query_prompt"] = self.query_prompts.get(dataset_name, None)
        if self.corpus_prompts is not None:
            ir_evaluator_kwargs["corpus_prompt"] = self.corpus_prompts.get(dataset_name, None)

        return evaluation.PyLateInformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=dict(corpus_items),
            relevant_docs=qrels_dict,
            name=self._get_human_readable_name(dataset_name, language),
            **ir_evaluator_kwargs,
        )

    def _validate_dataset_names(self):
        if not self.dataset_names:
            raise ValueError("dataset_names cannot be empty. Use None for all datasets.")
        missing = [n for n in self.dataset_names if n.lower() not in self.DATASETS]
        if missing:
            raise ValueError(f"Dataset(s) {missing} not found. Valid: {list(self.DATASETS)}")

    def _validate_languages(self):
        if not self.languages:
            raise ValueError("languages cannot be empty. Use None for all supported languages.")
        unsupported = [l for l in self.languages if l not in self.SUPPORTED_LANGUAGES]
        if unsupported:
            raise ValueError(f"Language(s) {unsupported} not supported. Supported: {self.SUPPORTED_LANGUAGES}")

    def _validate_prompts(self):
        for attr in ["query_prompts", "corpus_prompts"]:
            prompts = getattr(self, attr)
            if isinstance(prompts, str):
                setattr(self, attr, {name: prompts for name in self.dataset_names})
            elif prompts:
                missing = [n for n in self.dataset_names if n not in prompts]
                if missing:
                    raise ValueError(f"Missing {attr} for: {missing}")


class MultinomialBatchSampler(MultiDatasetDefaultBatchSampler):
    """Pick each batch's dataset with probability proportional to len(dataset) ** alpha.

    alpha=1.0 samples proportionally to dataset size, 0.5 smooths towards the small datasets,
    0 is uniform over datasets.
    """

    def __init__(self, dataset, batch_samplers, alpha: float = 1.0, generator=None, seed: int = 0):
        super().__init__(dataset, batch_samplers, generator, seed)
        weights = [len(dataset) ** alpha for dataset in self.dataset.datasets]
        self.sampling_probs = [weight / sum(weights) for weight in weights]

    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        offsets = [0] + list(itertools.accumulate(len(dataset) for dataset in self.dataset.datasets))
        iterators = [iter(sampler) for sampler in self.batch_samplers]
        resets = [0] * len(self.batch_samplers)
        probs = torch.tensor(self.sampling_probs, dtype=torch.float32)

        emitted = 0
        while emitted < len(self) and float(probs.sum()) > 0.0:
            index = torch.multinomial(probs, num_samples=1, replacement=True, generator=self.generator).item()
            batch = next(iterators[index], None)
            if batch is None:  # exhausted dataset: reseed and restart it so it keeps contributing
                resets[index] += 1
                iterators[index] = self._restart(index, resets[index])
                batch = next(iterators[index], None)
            if batch is None:
                logger.warning("MultinomialBatchSampler: dataset index %d yields no batches, excluding it.", index)
                probs[index] = 0.0
                continue
            emitted += 1
            yield [i + offsets[index] for i in batch]

    def _restart(self, index: int, reset_count: int):
        """Reseed the underlying sampler so a restarted dataset yields a fresh permutation."""
        sampler = self.batch_samplers[index]
        inner = getattr(sampler, "sampler", None)
        if hasattr(inner, "generator"):
            inner.generator.manual_seed(self.seed + self.epoch * 100000 + reset_count)
        elif hasattr(inner, "set_epoch"):
            inner.set_epoch(self.epoch * 100000 + reset_count)
        return iter(sampler)

    def __len__(self) -> int:
        return sum(len(sampler) for sampler in self.batch_samplers)


class StopAtStepCallback(TrainerCallback):
    def __init__(self, stop_at_step: int):
        self.stop_at_step = stop_at_step

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step >= self.stop_at_step:
            print(f"\n Reached target step {self.stop_at_step}. Stopping training...")
            control.should_training_stop = True
        return control


class KDToContrastive:
    """Converts the HF datasets with reranking scores into a contrastive one with query-positive-negatives format.

    Rows keep the teacher's reranker scores of the kept documents, in the same order as the
    positive and negative columns, so the collator can build the KL-div targets.
    """

    def __init__(
        self,
        queries: Dataset,
        documents: Dataset,
        num_negatives: int = 10,
        nv_threshold: float = 0.95,
    ) -> None:
        self.queries = dict(zip(queries["query_id"], queries["query"]))
        self.documents = dict(zip(documents["document_id"], documents["document"]))
        self.num_negatives = num_negatives
        self.nv_threshold = nv_threshold

    def negative_indices(self, example) -> list[int]:
        """Documents scored clearly below the positive and with a non-empty text, capped at num_negatives."""
        threshold = self.nv_threshold * example["scores"][0]
        return [
            i
            for i in range(1, len(example["document_ids"]))
            if example["scores"][i] < threshold and self.documents.get(example["document_ids"][i], "").strip()
        ][: self.num_negatives]

    def has_enough_negatives(self, example) -> bool:
        return bool(
            example["document_ids"]
            and self.queries.get(example["query_id"], "").strip()
            and self.documents.get(example["document_ids"][0], "").strip()
            and len(self.negative_indices(example)) >= self.num_negatives
        )

    def map_to_query_positive_negatives(self, example) -> dict[str, Any]:
        negatives = self.negative_indices(example)
        return {
            "query": self.queries[example["query_id"]],
            "positive": self.documents[example["document_ids"][0]],
            "teacher_scores": [float(example["rerank_scores"][i]) for i in [0, *negatives]],
            **{f"negative_{n}": self.documents[example["document_ids"][i]] for n, i in enumerate(negatives)},
        }


def load_train_datasets(cache_dir: str) -> DatasetDict:
    """Build/load the code and multilingual reranked KD splits, one cached Dataset per split.

    Every split shares the row schema (query, positive, negative_0..k, teacher_scores), so one
    builder covers them all and only the source repo differs: one repo per language for the
    <dataset>_<language> splits, one for code, one for the code-edit splits.
    """
    languages = ["en", "de", "it", "pt", "ar", "fr", "es", "sv", "no"]
    code_splits = [
        "apps", "synthetictext2sql", "cosqa", "codefeedbackst", "codefeedbackmt",
        "stackoverflowqa", "codetranscontest", "codetransdl",
        *[f"CodeSearchNet{ccr}_{language}" for language in ["go", "java", "javascript", "php", "python", "ruby"] for ccr in ["", "_ccr"]],
        *[f"CodeEditSearch_{language}" for language in ["c", "cpp", "go", "java", "javascript", "php", "python", "ruby", "rust", "scala", "shell", "swift", "typescript"]],
    ]
    multilingual_splits = [
        *[f"{split}_{language}" for split in ["hotpotqa", "nq", "msmarco", "fever", "squadv2", "fiqa", "trivia"] for language in languages],
        *[f"miracl_{language}" for language in ["ar", "en", "es", "fr"]],
        *[f"mldr_{language}" for language in ["en", "de", "es", "fr", "it", "pt", "ar"]],
    ]

    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = DatasetDict()
    for split, is_multilingual in [*[(s, False) for s in code_splits], *[(s, True) for s in multilingual_splits]]:
        try:
            dataset = Dataset.load_from_disk(f"{cache_dir}/{split}")
            if "teacher_scores" not in dataset.column_names:
                raise FileNotFoundError(f"cached {split} predates teacher_scores, rebuilding")
            print(f"Loaded dataset from disk: {split}")
            # Older caches carry a label column the KL-div collator would fight over
            if "label" in dataset.column_names:
                dataset = dataset.remove_columns("label")
        except FileNotFoundError:
            suffix = split.rsplit("_", 1)[-1] if is_multilingual else ("code-edit" if split.startswith("CodeEditSearch") else "code")
            repo = f"lightonai/embeddings-fine-tuning-filtered-{suffix}"
            print(f"Creating dataset: {split} (from {repo})")
            load = lambda config: load_dataset(repo, name=config, data_files=f"{config}/{split}-*", split="train", verification_mode="no_checks")
            scores = load("scores")
            # Baked into the cache — delete {cache_dir}/{split} after changing
            processor = KDToContrastive(load("queries"), load("documents"), num_negatives=10, nv_threshold=0.95)
            dataset = scores.filter(
                processor.has_enough_negatives,
                desc=f"Filtering examples with <10 negatives ({split})",
            ).map(
                processor.map_to_query_positive_negatives,
                remove_columns=scores.column_names,
                desc=f"Creating query-positive-negatives dataset ({split})",
            )
            print(f"  [{split}] {len(dataset):,}/{len(scores):,} rows kept, saving to disk")
            dataset.save_to_disk(f"{cache_dir}/{split}")
        train_dataset[split] = dataset
    return train_dataset


class ColBERTCollatorSampleNeg:
    """Collator for ColBERT that randomly samples a subset of negative columns per batch.

    ``teacher_scores`` is a per-row list aligned to ``[positive, negative_0, ...]``, so
    after sampling k negatives the matching scores are re-gathered in the sampled order
    and emitted as ``batch["label"]`` for the KL-div term.
    """

    def __init__(
        self,
        tokenize_fn: Callable,
        valid_label_columns: list[str] | None = None,
        num_negatives: int = 7,
    ) -> None:
        self.tokenize_fn = tokenize_fn
        self.num_negatives = num_negatives
        self.valid_label_columns = valid_label_columns or ["teacher_scores", "label", "scores"]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {"return_loss": True}
        columns = list(features[0].keys())

        if "dataset_name" in columns:
            columns.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        # detected but not materialized yet: the scores must follow the sampled order
        label_column = next(
            (c for c in self.valid_label_columns if c in columns), None
        )
        if label_column is not None:
            columns.remove(label_column)

        negative_columns = [col for col in columns if col.startswith("negative_")]
        other_columns = [col for col in columns if not col.startswith("negative_")]

        if self.num_negatives is not None and negative_columns:
            k = min(self.num_negatives, len(negative_columns))
            sampled_negatives = random.sample(negative_columns, k)
        else:
            sampled_negatives = negative_columns
        columns_to_process = other_columns + sampled_negatives

        if label_column is not None:
            if isinstance(features[0][label_column], list):
                # negative_i -> teacher_scores[i + 1]; index 0 is the positive
                indices = [0] + [int(c.split("_")[1]) + 1 for c in sampled_negatives]
                batch["label"] = torch.tensor(
                    [[row[label_column][i] for i in indices] for row in features],
                    dtype=torch.float,
                )
            else:
                batch["label"] = torch.tensor([row[label_column] for row in features])

        for column in columns_to_process:
            is_query = "query" in column or "anchor" in column
            texts = [row[column] for row in features]
            if isinstance(texts[0], list):
                texts = list(itertools.chain(*texts))
            # pad=False: pad to the batch max, not to the full 8192 token budget
            tokenized = self.tokenize_fn(texts, is_query=is_query, pad=False)
            for key, value in tokenized.items():
                batch[f"{column}_{key}"] = value

        return batch


def main():
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=8))]
    )

    parser = argparse.ArgumentParser(
        description="Fine-tune ColBERT with contrastive + KL-div distillation"
    )

    parser.add_argument("--model_name", type=str, default="lightonai/mLateOn-unsupervised")
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--query_length", type=int, default=8192)
    parser.add_argument("--document_length", type=int, default=8192)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--mini_batch_size", type=int, default=16)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--kldiv_weight", type=float, default=1.0)
    parser.add_argument("--teacher_temperature", type=float, default=0.1)
    parser.add_argument("--student_temperature", type=float, default=0.001)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dataset sampling exponent: probability proportional to size ** alpha (1.0 = proportional, 0 = uniform)",
    )
    parser.add_argument(
        "--stop_at_step",
        type=int,
        default=-1,
        help="Stop training after this many steps (set to -1 to disable)",
    )
    parser.add_argument("--eval_steps", type=int, default=40_000)
    parser.add_argument("--save_steps", type=int, default=20_000)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_on_start", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument(
        "--datasets_cache_dir",
        type=str,
        default="./cache/multilingual",
        help="Cache root: one subdirectory per built split, plus the HF hub/datasets caches",
    )

    args = parser.parse_args()

    # Build/cache the datasets on the main process only; other ranks wait, then load the cached version from disk
    with accelerator.main_process_first():
        train_dataset = load_train_datasets(cache_dir=args.datasets_cache_dir)
    print(train_dataset)

    model_shortname = args.model_name.rstrip("/").split("/")[-1]
    if model_shortname.startswith("checkpoint"):  # checkpoints are named after their run, not themselves
        model_shortname = args.model_name.rstrip("/").split("/")[-2]

    lr_str = f"{args.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e")
    temp_str = str(args.temperature).replace(".", "")
    run_name = (
        f"ColBERT-{model_shortname}-finetune-"
        f"lr{lr_str}-temp{temp_str}-"
        f"bs{args.batch_size}-"
        f"nv-retriever-0.95-10negs-7sampled-"
        f"contrast{args.contrastive_weight}-kl{args.kldiv_weight}-"
        f"ttemp{args.teacher_temperature}-stemp{args.student_temperature}-"
        f"-alpha{args.alpha}"
    )

    output_dir = f"{args.output_path}/{model_shortname}/{run_name}"

    print(f"\n{'=' * 60}")
    print("Training Configuration:")
    print(f"{'=' * 60}")
    print(f"Model: {args.model_name}")
    print(f"Datasets Cache Dir: {args.datasets_cache_dir} ({len(train_dataset)} splits)")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print(f"Query Length: {args.query_length}")
    print(f"Document Length: {args.document_length}")
    print(f"Batch Size: {args.batch_size}")
    print(f"GradCache Mini Batch Size: {args.mini_batch_size}")
    print(f"Contrastive weight: {args.contrastive_weight}")
    print(f"KL-div weight: {args.kldiv_weight}")
    print(f"Teacher temperature: {args.teacher_temperature}")
    print(f"Student temperature: {args.student_temperature}")
    print(f"Dataset Sampling: alpha={args.alpha} (1.0 = proportional to size, 0 = uniform)")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Stop at Step: {args.stop_at_step if args.stop_at_step > 0 else 'Disabled'}")
    print(f"Eval Batch Size: {args.eval_batch_size}")
    print(f"Eval on Start: {args.eval_on_start}")
    print(f"Resume From Checkpoint: {args.resume_from_checkpoint or '<none>'}")
    print(f"Run Name: {run_name}")
    print(f"Output Dir: {output_dir}")
    print(f"{'=' * 60}\n")

    # do_query_expansion=False: MASK-padding every query up to query_length would dominate the batch
    model = models.ColBERT(
        args.model_name,
        query_length=args.query_length,
        document_length=args.document_length,
        do_query_expansion=False,
        skiplist_words=[],
        model_kwargs={"attn_implementation": "flash_attention_2", "dtype": torch.float32},
    )

    dev_evaluator = MultilingualNanoBEIREvaluator(batch_size=args.eval_batch_size)
    train_loss = CachedContrastiveKLDiv(
        model=model,
        contrastive_temperature=args.temperature,
        student_temperature=args.student_temperature,
        teacher_temperature=args.teacher_temperature,
        contrastive_weight=args.contrastive_weight,
        kldiv_weight=args.kldiv_weight,
        mini_batch_size=args.mini_batch_size,
    )

    callbacks = []
    if args.stop_at_step > 0:
        callbacks.append(StopAtStepCallback(stop_at_step=args.stop_at_step))

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,  # global batch: split_batches=True
        per_device_eval_batch_size=args.eval_batch_size,
        multi_dataset_batch_sampler=partial(MultinomialBatchSampler, alpha=args.alpha),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_on_start=args.eval_on_start,
        save_steps=args.save_steps,
        logging_steps=1,
        fp16=False,
        bf16=True,
        seed=42,
        report_to="wandb",
        run_name=run_name,
        learning_rate=args.learning_rate,
        dataloader_num_workers=8,
        accelerator_config={
            "split_batches": True,
        },
    )

    data_collator = ColBERTCollatorSampleNeg(
        tokenize_fn=model.tokenize,
        num_negatives=7,  # sampled per step from the stored 10
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        callbacks=callbacks if callbacks else None,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(f"{output_dir}/final")

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Model saved to: {output_dir}/final")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
