import requests
from PIL import Image
import transformers
import torch

# 画像を参照
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# モデルとプロセッサの準備
model = transformers.LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16, low_cpu_mem_usage=True
)
processor = transformers.AutoProcessor.from_pretrained(
    "llava-hf/llava-1.5-7b-hf"
)