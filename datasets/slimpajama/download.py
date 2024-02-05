# partially download slim-pajama datasetc from huggingface
import requests

url_base = "https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train/chunk1/"
for i in range(100):
    url = url_base + f"example_train_{i}.jsonl.zst?download=true"
    r = requests.get(url, allow_redirects=True)
    with open(f"example_train_{i}.jsonl.zst", "wb") as f:
        f.write(r.content)
    if (i + 1) % 10 == 0:
        print(f"Downloaded {i} files")
