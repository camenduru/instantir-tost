import os, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler
from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler
from module.ip_adapter.utils import load_adapter_to_pipe
from pipelines.sdxl_instantir import InstantIRPipeline

def resize_img(input_image, max_side=1024, min_side=768, width=None, height=None, pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if width > 0 and height > 0:
        out_w, out_h = width, height
    elif width > 0:
        out_w = width
        out_h = round(h * width / w)
    elif height > 0:
        out_h = height
        out_w = round(w * height / h)
    else:
        out_w, out_h = w, h
    # Resize input to runtime size
    w, h = out_w, out_h
    if min(w, h) < min_side:
        ratio = min_side / min(w, h)
        w, h = round(ratio * w), round(ratio * h)
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        w, h = round(ratio * w), round(ratio * h)
    # Resize to cope with UNet and VAE operations
    w_resize_new = (w // base_pixel_number) * base_pixel_number
    h_resize_new = (h // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image, (out_w, out_h)

device = "cuda"
sdxl_repo_id = "/content/sd_xl_base_1.0"
dinov2_repo_id = "/content/dinov2"
lcm_repo_id = "/content/lcm"
PROMPT = "Photorealistic, highly detailed, hyper detailed photo - realistic maximum detail, 32k, \
ultra HD, extreme meticulous detailing, skin pore detailing, \
hyper sharpness, perfect without deformations, \
taken using a Canon EOS R camera, Cinematic, High Contrast, Color Grading. "
NEG_PROMPT = "blurry, out of focus, unclear, depth of field, over-smooth, \
sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, \
dirty, messy, worst quality, low quality, frames, painting, illustration, drawing, art, \
watermark, signature, jpeg artifacts, deformed, lowres"

with torch.inference_mode():
    torch_dtype = torch.bfloat16
    pipe = InstantIRPipeline.from_pretrained(sdxl_repo_id, torch_dtype=torch_dtype)
    load_adapter_to_pipe(pipe, "/content/InstantIR/adapter.pt", dinov2_repo_id)
    lora_alpha = pipe.prepare_previewers("/content/InstantIR")
    lora_alpha = pipe.prepare_previewers(lcm_repo_id, use_lcm=True)
    pipe.to(device=device, dtype=torch_dtype)
    pipe.scheduler = DDPMScheduler.from_pretrained(sdxl_repo_id, subfolder="scheduler")
    lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDPMScheduler.from_pretrained(sdxl_repo_id,subfolder="scheduler")
    lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)
    aggregator_state_dict = torch.load("/content/InstantIR/aggregator.pt",map_location="cpu")
    pipe.aggregator.load_state_dict(aggregator_state_dict)
    pipe.aggregator.to(device=device, dtype=torch_dtype)

def instantir_restore(lq, prompt="", negative_prompt="", steps=30, cfg_scale=7.0, guidance_end=1.0, creative_restoration=False, seed=3407, height=None, width=None, preview_start=0.0):
    if creative_restoration:
        if "lcm" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('lcm')
    else:
        if "previewer" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('previewer')

    if isinstance(guidance_end, int):
        guidance_end = guidance_end / steps
    elif guidance_end > 1.0:
        guidance_end = guidance_end / steps
    if isinstance(preview_start, int):
        preview_start = preview_start / steps
    elif preview_start > 1.0:
        preview_start = preview_start / steps

    lq, out_size = resize_img(lq, width=width, height=height)
    lq = [lq]
    generator = torch.Generator(device=device).manual_seed(seed)
    timesteps = [i * (1000//steps) + pipe.scheduler.config.steps_offset for i in range(0, steps)]
    timesteps = timesteps[::-1]

    prompt = PROMPT if len(prompt)==0 else prompt
    neg_prompt = NEG_PROMPT if len(negative_prompt)==0 else negative_prompt

    out = pipe(
        prompt=[prompt]*len(lq),
        image=lq,
        num_inference_steps=steps,
        generator=generator,
        timesteps=timesteps,
        negative_prompt=[neg_prompt]*len(lq),
        guidance_scale=cfg_scale,
        control_guidance_end=guidance_end,
        preview_start=preview_start,
        previewer_scheduler=lcm_scheduler,
        return_dict=False,
        save_preview_row=False,
    )
    out[0][0] = out[0][0].resize([out_size[0], out_size[1]], Image.BILINEAR)
    return out[0][0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image=values['input_image_check']
    input_image=download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='input_image')
    prompt = values['prompt']
    negative_prompt = values['negative_prompt']
    seed = values['seed']
    steps = values['steps']
    cfg = values['cfg']
    creative_restoration = values['creative_restoration']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    input_image = Image.open(input_image)
    lq_img = resize_img(input_image, max_side=1024, min_side=768, width=0, height=0, pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64)[0]
    output_image = instantir_restore(lq_img, prompt=prompt, negative_prompt=negative_prompt, seed=seed, steps=steps, cfg_scale=cfg, creative_restoration=creative_restoration, width=lq_img.size[0],  height=lq_img.size[1])
    output_image.save(f"/content/instantir-{seed}-tost.png")

    result = f"/content/instantir-{seed}-tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})
