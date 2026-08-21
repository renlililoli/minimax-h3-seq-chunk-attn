from __future__ import annotations

import contextlib
import copy
import types

import torch
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE


STATE_KEY = "_minimax_h3_seqattn_vae_controller"


class MiniMaxH3VAEController:
    def __init__(self, vae, *, tile_size: int, workspace_mib: int):
        self.vae = vae
        self.original_encode = vae.encode
        self.original_decode = vae.decode
        self.original_tile_size = self.model.tile_size
        self.configure(tile_size=tile_size, workspace_mib=workspace_mib)

    @property
    def model(self) -> MiniMaxH3VideoVAE:
        return self.vae.first_stage_model

    def configure(self, *, tile_size: int, workspace_mib: int) -> None:
        tile_size = int(tile_size)
        workspace_mib = int(workspace_mib)
        if tile_size < 128 or tile_size % self.model.vae_ratio:
            raise ValueError(
                "MiniMax H3 VAE tile_size must be at least 128 and divisible "
                f"by the spatial ratio {self.model.vae_ratio}"
            )
        if workspace_mib < 256:
            raise ValueError("MiniMax H3 VAE workspace_mib must be at least 256")
        self.tile_size = tile_size
        self.workspace_mib = workspace_mib

    def install(self) -> None:
        self.vae.encode = self.encode
        self.vae.decode = self.decode
        setattr(self.vae, STATE_KEY, self)

    def restore(self) -> None:
        self.vae.encode = self.original_encode
        self.vae.decode = self.original_decode
        self.model.tile_size = self.original_tile_size
        if getattr(self.vae, STATE_KEY, None) is self:
            delattr(self.vae, STATE_KEY)

    @contextlib.contextmanager
    def _configured_tile_size(self):
        original_tile_size = self.model.tile_size
        self.model.tile_size = self.tile_size
        try:
            yield
        finally:
            self.model.tile_size = original_tile_size

    def _load(self) -> None:
        import comfy.model_management as model_management

        model_management.load_models_gpu(
            [self.vae.patcher],
            memory_required=self.workspace_mib * 2**20,
            force_full_load=self.vae.disable_offload,
        )

    @torch.inference_mode()
    def encode(self, pixel_samples: torch.Tensor) -> torch.Tensor:
        import comfy.model_management as model_management

        pixels = self.vae.vae_encode_crop_pixels(pixel_samples).movedim(-1, 1)
        if pixels.ndim < 5:
            pixels = pixels.movedim(1, 0).unsqueeze(0)
        x = self.vae.process_input(pixels).to(self.vae.vae_dtype)
        self._load()

        with self._configured_tile_size(), model_management.cuda_device_context(
            self.vae.device
        ):
            if x.shape[2] == 1:
                encoded = self.model.encode(x.to(self.vae.device))
                return encoded.to(
                    device=self.vae.output_device,
                    dtype=self.vae.vae_output_dtype(),
                    copy=True,
                )

            clip_length = int(self.model.clip_length)
            if x.shape[2] % clip_length:
                pad_size = (-x.shape[2]) % clip_length
                x = torch.cat(
                    [x, x[:, :, -1:].repeat(1, 1, pad_size, 1, 1)], dim=2
                )

            moments = []
            for start in range(0, x.shape[2], clip_length):
                clip = x[:, :, start : start + clip_length].to(self.vae.device)
                clip.add_(1.0).mul_(0.5)
                clip.sub_(self.model.pixel_mean.to(clip)).div_(
                    self.model.pixel_std.to(clip)
                )
                encoded = self.model._adaptive_encode(clip)
                moments.append(encoded.to(self.vae.output_device, copy=True))
                del clip, encoded
                model_management.soft_empty_cache()

        moments = torch.cat(moments, dim=2)
        if self.model.token_drop > 0:
            moments = moments[:, :, : -self.model.token_drop]
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        latent_mean = self.model.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        latent_std = self.model.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return ((mean - latent_mean) / latent_std).to(
            dtype=self.vae.vae_output_dtype()
        )

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        import comfy.model_management as model_management

        if latent.ndim != 5 or latent.shape[0] != 1:
            raise ValueError("MiniMax H3 streamed VAE decode requires batch size 1")
        self._load()
        if latent.shape[2] == 1:
            with self._configured_tile_size(), model_management.cuda_device_context(
                self.vae.device
            ):
                decoded = self.model.decode(latent.to(self.vae.device))
            return self.vae.process_output(decoded).to(
                self.vae.output_device, copy=True
            ).movedim(1, -1)

        latent_mean = self.model.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        latent_std = self.model.latents_std.view(1, -1, 1, 1, 1).to(latent)
        z = latent * latent_std + latent_mean

        chunk_dec = self.model.tokens_chunk_size * self.model.vae_ratio_t
        split_count = int(self.model.token_drop > 0) + 1
        pseudo_total = z.shape[2] + self.model.token_drop
        pad_tokens = (-pseudo_total) % self.model.tokens_chunk_size
        pseudo_total += pad_tokens
        num_chunks = pseudo_total // self.model.tokens_chunk_size - int(
            self.model.token_drop > 0
        )
        if num_chunks < 1:
            pad_tokens += self.model.tokens_chunk_size
            num_chunks += 1
        if pad_tokens:
            z = torch.cat(
                [z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2
            )

        output_frames = self.model._decode_temporal_frame_plan(
            z.shape[2], num_chunks, pad_tokens
        )
        output = None
        overlap = None
        write_pos = 0

        def write_part(part: torch.Tensor) -> None:
            nonlocal output, write_pos
            frame_count = min(int(part.shape[2]), output_frames - write_pos)
            if frame_count <= 0:
                return
            if output is None:
                output = torch.empty(
                    (1, 3, output_frames, part.shape[-2], part.shape[-1]),
                    dtype=torch.float32,
                    device=self.vae.output_device,
                )
            pixels = part[:, :, :frame_count].float()
            pixels.mul_(self.model.pixel_std.to(pixels)).add_(
                self.model.pixel_mean.to(pixels)
            )
            pixels.clamp_(0.0, 1.0)
            output[:, :, write_pos : write_pos + frame_count].copy_(
                pixels.to(self.vae.output_device)
            )
            write_pos += frame_count

        with self._configured_tile_size(), model_management.cuda_device_context(
            self.vae.device
        ):
            for index in range(num_chunks):
                start = index * self.model.tokens_chunk_size
                stop = start + self.model.tokens_chunk_size + self.model.token_overlap
                clip_z = z[:, :, start:stop].to(
                    device=self.vae.device, dtype=self.vae.vae_dtype
                )
                clip_dec = self.model._adaptive_decode(clip_z)
                for split in range(split_count):
                    frame_start = split * chunk_dec
                    frame_stop = min(frame_start + chunk_dec, clip_dec.shape[2])
                    part = clip_dec[:, :, frame_start:frame_stop]
                    part = part[:, :, self.model.frame_pre_padding :]
                    if split == 0:
                        if overlap is not None:
                            part = self.model.blend(
                                overlap, part, self.model.frame_overlap, dim=-3
                            )
                            overlap = None
                        write_part(part)
                    else:
                        overlap = part.contiguous()
                if index == num_chunks - 1 and overlap is not None:
                    write_part(overlap)
                    overlap = None
                del clip_dec, clip_z
                model_management.soft_empty_cache()

        if output is None or write_pos != output_frames:
            raise RuntimeError(
                f"streamed VAE decode wrote {write_pos} of {output_frames} frames"
            )
        return output.movedim(1, -1)


def _rebind_wrapper_method(method, source, target):
    if isinstance(method, types.MethodType) and method.__self__ is source:
        return types.MethodType(method.__func__, target)
    return method


def _clone_vae_wrapper(vae):
    controller = getattr(vae, STATE_KEY, None)
    if isinstance(controller, MiniMaxH3VAEController):
        encode = controller.original_encode
        decode = controller.original_decode
    else:
        encode = vae.encode
        decode = vae.decode

    cloned = copy.copy(vae)
    patcher = getattr(vae, "patcher", None)
    if patcher is not None:
        cloned.patcher = patcher.clone()
        cloned.first_stage_model = cloned.patcher.model
    cloned.encode = _rebind_wrapper_method(encode, vae, cloned)
    cloned.decode = _rebind_wrapper_method(decode, vae, cloned)
    if hasattr(cloned, STATE_KEY):
        delattr(cloned, STATE_KEY)
    return cloned


def patch_minimax_h3_video_vae(vae, *, tile_size: int, workspace_mib: int):
    model = getattr(vae, "first_stage_model", None)
    if not isinstance(model, MiniMaxH3VideoVAE):
        actual = type(model).__name__ if model is not None else "None"
        raise TypeError(
            "MiniMaxH3VAEStreaming requires the native MiniMax H3 video VAE; "
            f"received {actual}"
        )
    patched = _clone_vae_wrapper(vae)
    controller = MiniMaxH3VAEController(
        patched, tile_size=tile_size, workspace_mib=workspace_mib
    )
    controller.install()
    return patched


def unpatch_minimax_h3_video_vae(vae):
    controller = getattr(vae, STATE_KEY, None)
    if isinstance(controller, MiniMaxH3VAEController):
        controller.restore()
    return vae


__all__ = [
    "MiniMaxH3VAEController",
    "patch_minimax_h3_video_vae",
    "unpatch_minimax_h3_video_vae",
]
