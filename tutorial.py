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

# プロンプトを準備
prompt = """USER: <image>
これは何の画像ですか。
ASSISTANT:
"""

# プロセッサとモデルで推論
inputs = processor(
    prompt,
    image,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=256 
)

# 結果を出力
output = outputs[0]

print(processor.decode(
    output,
    skip_special_tokens=True
))