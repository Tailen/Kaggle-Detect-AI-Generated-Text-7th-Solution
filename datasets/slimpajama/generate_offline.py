import numpy as np
import pandas as pd
import pickle as pkl
from vllm import LLM, SamplingParams

# load dataset
dataset_path = "./slimpajama_seeds.csv"
pajama_df = pd.read_csv(dataset_path)
df_batches = np.array_split(pajama_df, 250)

# parameter list
temperature = [0.9, 1.1]
top_p = [0.8, 0.85, 0.90, 0.95, 0.98]
frequency_penalty = [0.15, 0.3]
# get all combinations of parameters as tuples
param_tuples = [
    (t, p, f) for t in temperature for p in top_p for f in frequency_penalty
]
print(f"Number of parameter combinations: {len(param_tuples)}")

# generate completions
llm = LLM(model="HuggingFaceH4/zephyr-7b-beta", max_model_len=2048)
for i, df in enumerate(df_batches):
    if i >= 30 and i < 50:
        t, p, f = param_tuples[i - 30]
        print(f"Processing batch {i+1} with parameters: {t}, {p}, {f}")
        sampling_params = SamplingParams(
            temperature=t, top_p=p, frequency_penalty=f, max_tokens=2000
        )
        outputs = llm.generate(df["seed"].to_list(), sampling_params, use_tqdm=True)
        # save generated texts
        generated_texts = []
        reasons = []
        for output in outputs:
            generated_texts.append(output.outputs[0].text)
            reasons.append(output.outputs[0].finish_reason)
        df = pd.DataFrame({"text": generated_texts, "reason": reasons})
        try:
            df.to_csv(f"./generated/generated_{i}.csv", index=False)
        except:
            pkl.dump(df, open(f"./generated/generated_{i}.pkl", "wb"))
