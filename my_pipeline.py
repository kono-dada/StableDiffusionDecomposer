from typing import Callable, Dict, List, Optional, Tuple, Union
from diffusers import StableDiffusionPipeline
import torch
from callback import QKVRecordCallback, ATTN_BLOCKS, QKV
from my_attn import prep_unet_attention


def replace_attn_map(unet, callback: QKVRecordCallback, attn_indice: List[int], i: int):
    r"""
    Replace the attn_map

    Args:
        unet (UNet2DConditionModel): The UNet2DConditionModel.
        callback (QKVRecordCallback): The callback.
        attn_indice (List[int]): The indices of the attentions to replace.
        i (int): The index of the timestep.
    """
    namelist = callback.attentions.keys()
    target_attn = [ATTN_BLOCKS[j]
                   for j in attn_indice if ATTN_BLOCKS[j] in namelist]
    for _name in target_attn:
        attn_map = callback.attentions[_name][i].a
        for name, module in unet.named_modules():
            if name == _name:
                module.processor.attn_map = attn_map


def replace_attn_query(unet, callback: QKVRecordCallback, attn_indice: List[int], i: int):
    namelist = callback.attentions.keys()
    target_attn = [ATTN_BLOCKS[j]
                   for j in attn_indice if ATTN_BLOCKS[j] in namelist]
    for _name in target_attn:
        query = callback.attentions[_name][i].q
        for name, module in unet.named_modules():
            if name == _name:
                module.processor.my_query = query


def replace_attn(unet, ref_attentions: Dict[str, QKV],  qkv: List[str]):
    for name, module in unet.named_modules():
        if name in ref_attentions:
            module.processor.replaced_qkv = {
                _qkv: ref_attentions[name][_qkv] for _qkv in qkv}


@torch.no_grad()
def run_with_attn_replacement(
    p: StableDiffusionPipeline,
    replaced_attn_indice: List[int] = [],
    end: int = -1,
    ref_callback: Optional[QKVRecordCallback] = None,
    prompt: Union[str, List[str]] = None,
    timesteps: List[int] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
):
    prep_unet_attention(p.unet)
    callback_on_step_end_tensor_inputs: List[str] = ["latents"]
    p._guidance_scale = guidance_scale
    p._clip_skip = clip_skip
    p._interrupt = False
    device = p._execution_device

    prompt_embeds, negative_prompt_embeds = p.encode_prompt(
        prompt,
        device,
        1,
        p.do_classifier_free_guidance,
        negative_prompt,
        clip_skip=p.clip_skip,
    )

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if p.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Prepare timesteps
    num_inference_steps = len(timesteps)
    p.scheduler.set_timesteps(num_inference_steps, device=device)

    # 7. Denoising loop
    p._num_timesteps = len(timesteps)
    with p.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if p.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if p.do_classifier_free_guidance else latents

            # replace the attention map
            if ref_callback is not None and t > end:
                replace_attn_query(p.unet, ref_callback,
                                   replaced_attn_indice, i)
                # replace_attn_map(p.unet, ref_callback, replaced_attn_indice, i)

            # predict the noise residual
            noise_pred = p.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            # perform guidance
            if p.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + p.guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = p.scheduler.step(
                noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(
                    p, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop(
                    "prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop(
                    "negative_prompt_embeds", negative_prompt_embeds)

            progress_bar.update()

    image = p.vae.decode(
        latents / p.vae.config.scaling_factor, return_dict=False)[0]

    image = p.image_processor.postprocess(image, output_type='pt')
    # Offload all models
    p.maybe_free_model_hooks()

    return image


class MutualAttention:
    def __init__(self, p: StableDiffusionPipeline) -> None:
        self.p = p

    def __call__(
        self,
        replaced_attn_indice: List[int] = [],
        replaced_qkv: List[str] = [],
        end: int = -1,
        prompt: Union[str, List[str]] = '',
        ref_prompt: Union[str, List[str]] = '',
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        ref_latents: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        bs: int = 1,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        callback_on_step_end_tensor_inputs: List[str] = ["latents"]
        self.p._guidance_scale = guidance_scale
        self.p._clip_skip = clip_skip
        self.p._interrupt = False
        device = self.p._execution_device

        prompt_embeds, negative_prompt_embeds = self.p.encode_prompt(
            prompt,
            device,
            bs,
            self.p.do_classifier_free_guidance,
            negative_prompt,
            clip_skip=self.p.clip_skip,
        )

        ref_prompt_embeds, ref_negative_prompt_embeds = self.p.encode_prompt(
            ref_prompt,
            device,
            bs,
            self.p.do_classifier_free_guidance,
            negative_prompt,
            clip_skip=self.p.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.p.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            ref_prompt_embeds = torch.cat(
                [ref_negative_prompt_embeds, ref_prompt_embeds])

        # 4. Prepare timesteps
        self.p.scheduler.set_timesteps(num_inference_steps, device=device)

        # 7. Denoising loop
        self.p._num_timesteps = num_inference_steps
        with self.p.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.p.scheduler.timesteps):
                """
                ref_attentions: {
                    'down_blocks.0.attentions.0.transformer_blocks.0.attn1': QKV,
                    ...
                }
                """
                ref_latents, ref_attentions = self.ref_latents_step(
                    ref_latents, ref_prompt_embeds, t, replaced_attn_indice)

                if t > end:
                    latents = self.tgt_latents_step(
                        latents, prompt_embeds, t, replaced_qkv, ref_attentions)
                else:
                    latents = self.tgt_latents_step(latents, prompt_embeds, t)

                progress_bar.update()

        image = self.p.vae.decode(
            latents / self.p.vae.config.scaling_factor, return_dict=False)[0]
        image_ref = self.p.vae.decode(
            ref_latents / self.p.vae.config.scaling_factor, return_dict=False)[0]

        image = self.p.image_processor.postprocess(image, output_type='pt')
        image_ref = self.p.image_processor.postprocess(
            image_ref, output_type='pt')
        # Offload all models
        self.p.maybe_free_model_hooks()

        return image, image_ref

    def ref_latents_step(
        self,
        ref_latents,
        ref_prompt_embeds,
        t,
        replaced_attn_indice: List[int] = [],
    ) -> Tuple[torch.FloatTensor, Dict[str, QKV]]:
        prep_unet_attention(self.p.unet)
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat(
            [ref_latents] * 2) if self.p.do_classifier_free_guidance else ref_latents

        # predict the noise residual
        noise_pred = self.p.unet(
            latent_model_input,
            t,
            encoder_hidden_states=ref_prompt_embeds,
            return_dict=False,
        )[0]

        # perform guidance
        if self.p.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.p.guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        ref_latents = self.p.scheduler.step(
            noise_pred, t, ref_latents, return_dict=False)[0]

        # record the required attentions
        modules = list(self.p.unet.named_modules())
        attentions = {}
        target_attn = [ATTN_BLOCKS[j] for j in replaced_attn_indice]
        for name, module in modules:
            if name in target_attn:
                q, k, v, a = module.processor.record_QKV()
                attentions[name] = QKV(
                    t, name, q, k, v, a, module.processor.guidance)

        return ref_latents, attentions

    def tgt_latents_step(
        self,
        latents,
        prompt_embeds,
        t,
        replaced_qkv: List[str] = [],
        ref_attentions: Dict[str, QKV] = {}
    ) -> torch.FloatTensor:
        prep_unet_attention(self.p.unet)
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat(
            [latents] * 2) if self.p.do_classifier_free_guidance else latents

        # replace the attention components
        replace_attn(self.p.unet, ref_attentions, replaced_qkv)

        # predict the noise residual
        noise_pred = self.p.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        # perform guidance
        if self.p.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.p.guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.p.scheduler.step(
            noise_pred, t, latents, return_dict=False)[0]

        return latents
