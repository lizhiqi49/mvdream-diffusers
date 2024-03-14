import torch
from mvdream.utils import get_camera
from mvdream.pipeline_mvdream import MVDreamPipeline

pipe = MVDreamPipeline.from_pretrained(
    "lzq49/mvdream-sd21-diffusers", torch_dtype=torch.float16, trust_remote_code=True
)
pipe.to("cuda")

c2ws = get_camera(4, 0, 0).cuda()

with torch.cuda.amp.autocast(dtype=torch.float16):
    images = pipe(
        prompt=["a cute teddy bear"],   # batch size = 1
        c2ws=c2ws.unsqueeze(0),         # batch size = 1
        height=256,
        width=256,
        output_type="np"
    ).images