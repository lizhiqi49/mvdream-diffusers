# mvdream-diffusers
Implementation of MVDream using huggingface's diffusers. 

The model checkpoint of version sd-v2.1 can be found at [this huggingface page](https://huggingface.co/lzq49/mvdream-sd21-diffusers).


## Usage

```py
import torch
from diffusers import MVDreamPipeline
from diffusers.utils.camera import get_camera

pipe = MVDreamPipeline.from_pretrained("lzq49/mvdream_sd21_diffusers")
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