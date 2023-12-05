# mvdream-diffusers
Implementation of MVDream using huggingface's diffusers


## Usage

```py
import torch
from diffusers import MVDreamPipeline
from diffusers.utils.camera import get_camera

pipe = MVDreamPipeline.from_pretrained("mvdream_sd21_diffusers")
pipe.to("cuda")

c2ws = get_camera(4, 0, 0).cuda()

with torch.cuda.amp.autocast(dtype=torch.float16):
    images = pipe(
        prompt="a cute teddy bear",
        c2ws=c2ws,
        height=256,
        width=256,
        output_type="np"
    ).images


```