import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers.models.autoencoders.autoencoder_asym_kl import AsymmetricAutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from typing import Union, Optional, Callable, Dict, List, Any
import numpy as np
import random

def insert_rand_word(sentence,word):
    sent_list = sentence.split(' ')
    sent_list.insert(random.randint(0, len(sent_list)), word)
    new_sent = ' '.join(sent_list)
    return new_sent

def prompt_augmentation(prompt, aug_style, tokenizer=None, repeat_num=2):
    if aug_style =='RNA':
        for i in range(repeat_num):
            randnum  = np.random.choice(100000)
            prompt = insert_rand_word(prompt,str(randnum))
    elif aug_style =='RWA':
        for i in range(repeat_num):
            randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
            prompt = insert_rand_word(prompt,randword)
    elif aug_style =='CWR':
        wordlist = prompt.split(" ")
        for i in range(repeat_num):
            randword = np.random.choice(wordlist)
            prompt = insert_rand_word(prompt,randword)
    else:
        raise Exception('This style of prompt augmnentation is not written')
    return prompt


@torch.no_grad()
def inpaint_image(
    pipe: StableDiffusionInpaintPipeline,
    prompt: Union[str, List[str]] = None,
    image: PipelineImageInput = None,
    mask_image: PipelineImageInput = None,
    masked_image_latents: torch.Tensor = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    padding_mask_crop: Optional[int] = None,
    strength: float = 1.0,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: int = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    gaussian_perturbation: float = 0.0,
    **kwargs,
):
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 0. Default height and width to unet
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    # 1. Check inputs
    pipe.check_inputs(
        prompt,
        image,
        mask_image,
        height,
        width,
        strength,
        callback_steps,
        output_type,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
        padding_mask_crop,
    )

    pipe._guidance_scale = guidance_scale
    pipe._clip_skip = clip_skip
    pipe._cross_attention_kwargs = cross_attention_kwargs
    pipe._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        pipe.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
        clip_skip=pipe.clip_skip,
    )
    if gaussian_perturbation != 0:
        prompt_embeds += torch.randn(prompt_embeds.size(), device=prompt_embeds.device) * gaussian_perturbation
    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            pipe.do_classifier_free_guidance,
        )

    # 4. set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, timesteps, sigmas
    )
    timesteps, num_inference_steps = pipe.get_timesteps(
        num_inference_steps=num_inference_steps, strength=strength, device=device
    )
    # check that number of inference steps is not < 1 - as this doesn't make sense
    if num_inference_steps < 1:
        raise ValueError(
            f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
            f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
        )
    # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
    is_strength_max = strength == 1.0

    # 5. Preprocess mask and image

    if padding_mask_crop is not None:
        crops_coords = pipe.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
        resize_mode = "fill"
    else:
        crops_coords = None
        resize_mode = "default"

    original_image = image
    init_image = pipe.image_processor.preprocess(
        image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
    )
    init_image = init_image.to(dtype=torch.float32)

    # 6. Prepare latent variables
    num_channels_latents = pipe.vae.config.latent_channels
    num_channels_unet = pipe.unet.config.in_channels
    return_image_latents = num_channels_unet == 4

    latents_outputs = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
        image=init_image,
        timestep=latent_timestep,
        is_strength_max=is_strength_max,
        return_noise=True,
        return_image_latents=return_image_latents,
    )

    if return_image_latents:
        latents, noise, image_latents = latents_outputs
    else:
        latents, noise = latents_outputs

    # 7. Prepare mask latent variables
    mask_condition = pipe.mask_processor.preprocess(
        mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
    )

    if masked_image_latents is None:
        masked_image = init_image * (mask_condition < 0.5)
    else:
        masked_image = masked_image_latents

    mask, masked_image_latents = pipe.prepare_mask_latents(
        mask_condition,
        masked_image,
        batch_size * num_images_per_prompt,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        pipe.do_classifier_free_guidance,
    )

    # 8. Check that sizes of mask, masked image and latents match
    if num_channels_unet == 9:
        # default case for runwayml/stable-diffusion-inpainting
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != pipe.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {pipe.unet.config} expects"
                f" {pipe.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )
    elif num_channels_unet != 4:
        raise ValueError(
            f"The unet {pipe.unet.__class__} should have either 4 or 9 input channels, not {pipe.unet.config.in_channels}."
        )

    # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    # 9.1 Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None
        else None
    )

    # 9.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 10. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    pipe._num_timesteps = len(timesteps)
    # with pipe.progress_bar(total=num_inference_steps) as progress_bar:
    for i, t in enumerate(timesteps):
        if pipe.interrupt:
            continue

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents

        # concat latents, mask, masked_image_latents in the channel dimension
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        if num_channels_unet == 9:
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

        # predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=pipe.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if pipe.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        if num_channels_unet == 4:
            init_latents_proper = image_latents
            if pipe.do_classifier_free_guidance:
                init_mask, _ = mask.chunk(2)
            else:
                init_mask = mask

            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                init_latents_proper = pipe.scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([noise_timestep])
                )

            latents = (1 - init_mask) * init_latents_proper + init_mask * latents

        if callback_on_step_end is not None:
            callback_kwargs = {}
            for k in callback_on_step_end_tensor_inputs:
                callback_kwargs[k] = locals()[k]
            callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)

            latents = callback_outputs.pop("latents", latents)
            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
            negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
            mask = callback_outputs.pop("mask", mask)
            masked_image_latents = callback_outputs.pop("masked_image_latents", masked_image_latents)
            

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
            # progress_bar.update()
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(pipe.scheduler, "order", 1)
                callback(step_idx, t, latents)

    if not output_type == "latent":
        condition_kwargs = {}
        if isinstance(pipe.vae, AsymmetricAutoencoderKL):
            init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
            init_image_condition = init_image.clone()
            init_image = pipe._encode_vae_image(init_image, generator=generator)
            mask_condition = mask_condition.to(device=device, dtype=masked_image_latents.dtype)
            condition_kwargs = {"image": init_image_condition, "mask": mask_condition}
        image = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator, **condition_kwargs
        )[0]
        image, has_nsfw_concept = pipe.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    if padding_mask_crop is not None:
        image = [pipe.image_processor.apply_overlay(mask_image, original_image, i, crops_coords) for i in image]

    # Offload all models
    pipe.maybe_free_model_hooks()

    if not return_dict:
        return image

    return image

def prepare_latents(pipe, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)
        return latents

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * self.scheduler.init_noise_sigma
    return latents


def partial_denoise(
    pipe: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_total_steps: int = 100,
    num_partial_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    **kwargs,
    ):
    timesteps, sigmas, device = None, None, pipe._execution_device
    pipe.scheduler.set_timesteps(num_total_steps)
    timesteps = pipe.scheduler.timesteps
    timesteps = timesteps[-num_partial_steps::].cpu()
    latents /= pipe.vae_scale_factor
    num_inference_steps = num_partial_steps
    
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 0. Default height and width to unet
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor
    # to deal with lora scaling and other possible forward hooks

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    pipe._guidance_scale = guidance_scale
    pipe._guidance_rescale = guidance_rescale
    pipe._clip_skip = clip_skip
    pipe._cross_attention_kwargs = cross_attention_kwargs
    pipe._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device

    # 3. Encode input prompt
    lora_scale = (
        pipe.cross_attention_kwargs.get("scale", None) if pipe.cross_attention_kwargs is not None else None
    )

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        pipe.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=pipe.clip_skip,
    )

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            pipe.do_classifier_free_guidance,
        )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, timesteps, sigmas
    )

    # 5. Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        else None
    )

    # 6.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    pipe._num_timesteps = len(timesteps)
    latents = pipe.scheduler.add_noise(latents, torch.randn_like(latents), timesteps[0:1])

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipe.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if pipe.do_classifier_free_guidance and pipe.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipe.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(pipe.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    if not output_type == "latent":
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        image, has_nsfw_concept = pipe.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload all models
    pipe.maybe_free_model_hooks()

    if not return_dict:
        return (image, has_nsfw_concept)

    return image[0]