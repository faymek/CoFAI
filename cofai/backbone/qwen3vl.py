"""
Backbone implementation for Qwen3VL using the transformers library.

This backbone exposes the Visual Encoder of Qwen3VL as a feature extraction /
re-injection interface for the CoFAI codec framework. Unlike the vision-only
backbones (DINOv2 / MAE / SigLIP2), Qwen3VL is a Vision-Language Model (VLM):
the intermediate visual ``hidden_states`` extracted at the mandatory breakpoint
(the first deepstack index, "slot 9") are the object to be compressed /
decompressed, after which generation continues through the rest of the visual
encoder and the language model.

Available models on the HuggingFace Hub:
   - 'Qwen/Qwen3-VL-8B-Instruct' (default)

Workflow:
   1. ``prepare_image``: preprocess a PIL image into ``pixel_values`` and
      ``image_grid_thw`` (expand/shrink to min/max pixels).
   2. ``extract_features`` (encode): run the visual encoder up to the first
      deepstack index and return token features ``(1, L, C)`` (native encoder
      layout ``(L, C)`` with batch dim added).
   ``encode_image`` combines steps 1–2 and also returns ``token_res`` ``(H, W)``.
   3. ``decode_text`` (decode): continue running the visual encoder
      to compute all features (including deepstack features), then run the
      language model to generate text.

Without any compression applied between step 2 and step 3, this backbone
replicates the exact behaviour of the original Qwen3VL in transformers.

NOTE: ``_pos_and_rotary`` adapts at runtime to the transformers visual API
change around v5.8, so this works on both <5.8 and >=5.8:
   - transformers < 5.8: visual instance methods
     ``Qwen3VLVisionModel.fast_pos_embed_interpolate`` and
     ``Qwen3VLVisionModel.rot_pos_emb``.
   - transformers >= 5.8: standalone helpers
     ``transformers.vision_utils.get_vision_bilinear_indices_and_weights`` and
     ``transformers.vision_utils.get_vision_position_ids``.
Verified on the project's pinned transformers 5.7.0 (torch 2.4).
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
)


class Qwen3VLBackbone(nn.Module):
    """
    Qwen3VL backbone using the transformers library.

    Extracts and re-injects intermediate visual features from a Qwen3VL model,
    serving as the standard VLM backbone API in the CoFAI framework. The
    ``slot`` (breakpoint) is fixed to the first deepstack index of the visual
    encoder ("slot 9"), which is the mandatory breakpoint for this model.

    The encode part (``extract_features``) runs ``visual.blocks[:slot]`` and
    returns token features ``(1, L, C)``. The decode part
    (``decode_text``) accepts the same layout (with ``token_res``),
    runs the remaining blocks, collects deepstack features, and drives the
    language model to generate text.

    Args:
        model_path (str): HuggingFace model name or local path.
                          Defaults to 'Qwen/Qwen3-VL-8B-Instruct'.
        device (str): Device to run the model on. Defaults to "cuda" if available, else "cpu".
        default_prompt (str): Fallback prompt used when none is supplied. Defaults to "".
        local_files_only (bool): Whether to load strictly from local cache. Defaults to False.
        dtype (str or torch.dtype): Model weight dtype passed to ``from_pretrained``.
                          Accepts "auto", "bfloat16", "float16", etc., or a torch.dtype.
                          Defaults to "bfloat16".
        min_pixels (int, optional): Minimum pixels for image resizing in the processor.
                          Defaults to 768*28*28.
        max_pixels (int, optional): Maximum pixels for image resizing in the processor.
                          Defaults to 1536*28*28.
        max_new_tokens (int): Default max new tokens for generation. Defaults to 256.
        do_sample (bool): Default sampling flag for generation. Defaults to False.
        slot (int, optional): Encoder breakpoint, following timm slicing convention
                          (encode=blocks[:slot], decode=blocks[slot:]); "slot N" is
                          the feature after layer N-1, and slot may start from 0.
                          Defaults to the first deepstack feature (slot 9, after
                          layer 8), the latest valid breakpoint. A smaller slot is
                          allowed; a larger one is rejected.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        default_prompt: str = "",
        local_files_only: bool = False,
        dtype: Union[str, torch.dtype] = "bfloat16",
        min_pixels: Optional[int] = 768 * 28 * 28,
        max_pixels: Optional[int] = 1536 * 28 * 28,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        slot: Optional[int] = None,
    ):
        super().__init__()
        self.device = device
        self.default_prompt = default_prompt
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
            local_files_only=local_files_only,
            attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.image_token_id = self.model.config.image_token_id

        # Breakpoint slot. The first deepstack index (layer index 8) is the latest
        # Breakpoint slot, following timm slicing convention (like Dinov2TimmBackbone):
        #   encode part = blocks[:slot], decode part = blocks[slot:].
        # So "slot N" means the feature after layer index N-1. deepstack_visual_indexes[0]
        # is the first deepstack layer (e.g. 8), whose feature is the default
        # breakpoint "slot 9". That is the latest valid breakpoint: decode can only
        # re-collect deepstack features at/after the breakpoint, so a larger slot
        # would lose the first deepstack feature. A smaller slot (down to 0) is allowed.
        self.first_ds_idx = self.model.model.visual.deepstack_visual_indexes[0]
        max_slot = self.first_ds_idx + 1
        if slot is None:
            self.slot = max_slot
        else:
            if not 0 <= slot <= max_slot:
                raise ValueError(
                    f"slot={slot} is out of range [0, {max_slot}]. The upper bound "
                    f"{max_slot} is the first deepstack feature (after layer "
                    f"{self.first_ds_idx}); a larger slot would lose deepstack "
                    f"features that cannot be recovered during decode."
                )
            self.slot = slot

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #
    def _get_prompt_template(self, prompt: str):
        """Tokenize a text+image template without loading or processing an image.

        After codec, the original input feature is no longer available. This
        splits token ids around the image marker so reconstructed visual tokens
        can be inserted later.

        Args:
            prompt (str): The text prompt to wrap.

        Returns:
            template (dict): Prefix and suffix token ids.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.processor.tokenizer(
            formatted,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]

        img_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)[0]
        if len(img_positions) != 1:
            raise ValueError(
                "Qwen3VL prompt template must contain exactly one image placeholder, "
                f"got {len(img_positions)}."
            )

        img_start = img_positions[0].item()
        img_end = img_positions[-1].item() + 1

        template = {
            "prefix_ids": input_ids[:img_start],
            "suffix_ids": input_ids[img_end:],
        }
        return template

    def _build_inputs(self, prompt: str, num_visual_tokens: int):
        """Assemble language-model inputs with a reconstructed visual token span.

        Args:
            prompt (str): The text prompt.
            num_visual_tokens (int): Number of pooled visual tokens to reserve.

        Returns:
            inputs (dict): input_ids, attention_mask, mm_token_type_ids, prompt_len.
        """
        template = self._get_prompt_template(prompt)
        device = self.device

        prefix_ids = template["prefix_ids"].to(device)
        suffix_ids = template["suffix_ids"].to(device)
        img_ids = torch.full(
            (num_visual_tokens,), self.image_token_id, dtype=torch.long, device=device
        )
        input_ids = torch.cat([prefix_ids, img_ids, suffix_ids]).unsqueeze(0)

        prefix_mm = torch.zeros_like(prefix_ids, dtype=torch.int)
        suffix_mm = torch.zeros_like(suffix_ids, dtype=torch.int)
        img_mm = torch.ones(num_visual_tokens, dtype=torch.int, device=device)
        mm_token_type_ids = torch.cat([prefix_mm, img_mm, suffix_mm]).unsqueeze(0)

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "prompt_len": input_ids.shape[1],
        }

    def _pos_and_rotary(self, image_grid_thw: torch.Tensor):
        """Compute visual positional and rotary embeddings (version-robust).

        Handles the transformers API change around v5.8: newer versions expose
        standalone helpers in ``transformers.vision_utils``, while older ones
        (e.g. 5.7.x, the version this project pins) provide them as instance
        methods on the visual model.

        Args:
            image_grid_thw (torch.Tensor): Grid (t, h, w) for the image(s).

        Returns:
            pos_embeds (torch.Tensor): Interpolated positional embeddings, shape (seq_len, C).
            rotary_pos_emb (torch.Tensor): Rotary embeddings, shape (seq_len, dim).
        """
        visual = self.model.model.visual
        image_grid_thw = image_grid_thw.to(self.device)

        if hasattr(visual, "fast_pos_embed_interpolate") and hasattr(
            visual, "rot_pos_emb"
        ):
            # transformers < 5.8: instance-method API
            pos_embeds = visual.fast_pos_embed_interpolate(image_grid_thw)
            rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
        else:
            # transformers >= 5.8: standalone helpers in transformers.vision_utils
            from transformers.vision_utils import (  # type: ignore[import-not-found]
                get_vision_bilinear_indices_and_weights,
                get_vision_position_ids,
            )

            bilinear_indices, bilinear_weights = (
                get_vision_bilinear_indices_and_weights(
                    image_grid_thw,
                    num_grid_per_side=visual.num_grid_per_side,
                    spatial_merge_size=visual.spatial_merge_size,
                )
            )
            pos_embeds = (
                visual.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]
            ).sum(0)
            position_ids = get_vision_position_ids(
                image_grid_thw, visual.spatial_merge_size
            )
            rotary_pos_emb = visual.rotary_pos_emb(position_ids)

        return pos_embeds, rotary_pos_emb

    def _visual_attn_context(
        self, image_grid_thw: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Build ``cu_seqlens`` and rotary position embeddings for visual blocks."""
        device = self.device
        grid = image_grid_thw.to(device)
        cu_seqlens = torch.repeat_interleave(
            grid[:, 1] * grid[:, 2],
            grid[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(device)
        _, rotary_pos_emb = self._pos_and_rotary(grid)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        return cu_seqlens, (emb.cos(), emb.sin())

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def prepare_image(self, image: Union[str, Image.Image, torch.Tensor]):
        """Preprocess one image into ``pixel_values`` and ``image_grid_thw``.

        Args:
            image: Image path, PIL image, or CHW tensor. Batched tensors must
                contain exactly one image.

        Returns:
            pixel_values (torch.Tensor): Flattened patch pixel values.
            image_grid_thw (torch.Tensor): Grid (t, h, w) for the image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            if image.dim() == 4:
                if image.size(0) != 1:
                    raise ValueError(
                        "Qwen3VL feature evaluation currently requires batch size 1."
                    )
                image = image[0]
            image = image.detach().cpu()

        inputs = self.processor.image_processor(
            images=image,
            return_tensors="pt",
        )
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        return pixel_values, image_grid_thw

    def extract_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Encode: run the visual encoder up to the breakpoint slot.

        Runs ``visual.blocks[:slot]`` and returns encoder tokens in the same
        ``(B, L, C)`` layout used by :class:`~cofai.models.base.DinoFeatureCodecModel`
        (single-image eval uses ``B=1``, ``L=H*W``).

        Args:
            pixel_values (torch.Tensor): Patch pixel values from ``prepare_image``.
            image_grid_thw (torch.Tensor): Grid (t, h, w) from ``prepare_image``.

        Returns:
            hidden_states (torch.Tensor): Token features, shape ``(1, L, C)``.
        """
        visual = self.model.model.visual
        device = self.device

        block_dtype = next(visual.blocks[0].parameters()).dtype

        hidden_states = visual.patch_embed(pixel_values.to(block_dtype).to(device))

        # Build and apply positional embeddings (version-robust)
        pos_embeds, _ = self._pos_and_rotary(image_grid_thw)
        hidden_states = hidden_states + pos_embeds.to(device).to(block_dtype)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        cu_seqlens, position_embeddings = self._visual_attn_context(
            image_grid_thw, seq_len
        )

        # Encode part: run blocks[:slot] (timm slicing convention)
        for blk in visual.blocks[: self.slot]:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        return hidden_states.unsqueeze(0)

    def encode_image(
        self, image: Union[str, Image.Image, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess an image and extract breakpoint tokens plus grid shape.

        Convenience wrapper around ``prepare_image`` + ``extract_features``.

        Returns:
            h_tokens: Token features, shape ``(1, L, C)``.
            token_res: Token grid ``(H, W)`` with ``L = H * W``.
        """
        pixel_values, image_grid_thw = self.prepare_image(image)
        h_tokens = self.extract_features(pixel_values, image_grid_thw)
        token_res = (
            int(image_grid_thw[0, 1].item()),
            int(image_grid_thw[0, 2].item()),
        )
        return h_tokens, token_res

    def decode_text(
        self,
        hidden_states: torch.Tensor,
        prompt: Optional[str] = "",
        token_res: Optional[tuple[int, int]] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: float = 1.0,
        what_to_return: str = "string",  # "string" or "ids"
        **gen_kwargs,
    ):
        """Decode: continue the visual encoder from reconstructed features and generate text.

        Accepts codec tokens ``(1, L, C)`` or flat ``(L, C)``. ``token_res=(H, W)``
        is required so the visual grid metadata can be reconstructed.

        Args:
            hidden_states (torch.Tensor): Reconstructed tokens, ``(1, L, C)`` or ``(L, C)``.
            prompt (str, optional): The text prompt. Falls back to ``default_prompt`` if empty.
            token_res (tuple[int, int], optional): Token grid ``(H, W)`` with ``L=H*W``.
            max_new_tokens (int, optional): Max new tokens; falls back to the
                instance default ``self.max_new_tokens`` when None.
            do_sample (bool, optional): Sampling flag; falls back to the instance
                default ``self.do_sample`` when None.
            temperature (float): Sampling temperature. Defaults to 1.0.
            what_to_return (str): "string" for decoded text, "ids" for token ids.
            **gen_kwargs: Extra keyword arguments forwarded to ``model.generate``.

        Returns:
            str or torch.Tensor: Generated text, or generated token ids.
        """
        if not prompt:
            prompt = self.default_prompt
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if do_sample is None:
            do_sample = self.do_sample

        visual = self.model.model.visual
        device = self.device

        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
        if hidden_states.dim() != 2:
            raise ValueError(
                "decode_text expects token features (1, L, C) or (L, C), "
                f"got shape {tuple(hidden_states.shape)}"
            )
        if token_res is None:
            raise ValueError("token_res=(H, W) is required for decode_text")
        h, w = int(token_res[0]), int(token_res[1])
        if h * w != hidden_states.shape[0]:
            raise ValueError(
                f"token_res={token_res} implies L={h * w}, but hidden_states has L={hidden_states.shape[0]}"
            )

        image_grid_thw = torch.tensor([[1, h, w]], device=device, dtype=torch.int64)
        seq_len = h * w
        cu_seqlens, position_embeddings = self._visual_attn_context(
            image_grid_thw, seq_len
        )
        hidden_states = hidden_states.to(visual.dtype).to(device)

        # Decode part: run blocks[slot:] and collect deepstack features.
        deepstack_features = []  # 1st LM input: deepstack visual features

        # The input feature is the output of layer (slot-1). If that breakpoint
        # layer is itself a deepstack index, its feature == the input, so collect
        # it here without re-running the block.
        break_layer = self.slot - 1
        if break_layer in visual.deepstack_visual_indexes:
            idx = visual.deepstack_visual_indexes.index(break_layer)
            deepstack_features.append(visual.deepstack_merger_list[idx](hidden_states))

        for layer_num, blk in enumerate(visual.blocks[self.slot :], start=self.slot):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in visual.deepstack_visual_indexes:
                idx = visual.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(
                    visual.deepstack_merger_list[idx](hidden_states)
                )

        pooler_output = visual.merger(
            hidden_states
        )  # 2nd LM input: pooled visual features
        num_visual_tokens = pooler_output.shape[0]

        inputs = self._build_inputs(prompt, num_visual_tokens)

        # CORE: monkey-patch the visual forward so generation uses the
        # reconstructed visual features instead of recomputing them.
        original_visual_forward = visual.forward

        def patched_visual_forward(_hidden_states, grid_thw, **kwargs):
            return BaseModelOutputWithDeepstackFeatures(
                last_hidden_state=hidden_states,
                pooler_output=pooler_output,
                deepstack_features=deepstack_features,
            )

        visual.forward = patched_visual_forward

        dummy_pixel_values = torch.zeros(
            1, 3, 224, 224, device=self.device, dtype=visual.dtype
        )
        try:
            generation_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "mm_token_type_ids": inputs["mm_token_type_ids"],
                # Must pass a dummy. Otherwise the LM treats the task as
                # text-only and ignores the visual features.
                "pixel_values": dummy_pixel_values,
                "image_grid_thw": image_grid_thw,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "use_cache": True,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                **gen_kwargs,
            }

            with torch.no_grad():
                generated_ids = self.model.generate(**generation_inputs)

                new_ids = generated_ids[:, inputs["input_ids"].shape[1] :]

                output_text = self.processor.batch_decode(
                    new_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            if what_to_return == "ids":
                return generated_ids[0]
            return output_text

        finally:
            # Restore the vanilla visual forward function
            visual.forward = original_visual_forward
