---
datasets:
- tiiuae/falcon-refinedweb
language:
- en
- de
- es
- fr
inference: false
---

# 🚀 Falcon-40B

**Falcon-40B is a 40B parameters causal decoder-only model built by [TII](https://www.tii.ae) and trained on 1,000B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) enhanced with curated corpora. It is made available under the [TII Falcon LLM License](https://huggingface.co/tiiuae/falcon-40b/blob/main/LICENSE.txt).**

*Paper coming soon 😊.*

## Why use Falcon-40B?

* **It is the best open-source model currently available.** Falcon-40B outperforms [LLaMA](https://github.com/facebookresearch/llama), [StableLM](https://github.com/Stability-AI/StableLM), [RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1), [MPT](https://huggingface.co/mosaicml/mpt-7b), etc. See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
* **It features an architecture optimized for inference**, with FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) and multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)). 
* **It is made available under a license allowing commercial use**, see the details of the [TII Falcon LLM License](https://huggingface.co/tiiuae/falcon-40b/blob/main/LICENSE.txt) below.

⚠️ **This is a raw, pretrained model, which should be further finetuned for most usecases.** If you are looking for a version better suited to taking generic instructions in a chat format, we recommend taking a look at [Falcon-40B-Instruct](https://huggingface.co/tiiuae/falcon-40b-instruct). 

💸 **Looking for a smaller, less expensive model?** [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) is Falcon-40B's small brother!

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-40b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

```

💥 **Falcon LLMs require PyTorch 2.0 for use with `transformers`!**


# Model Card for Falcon-40B

## Model Details

### Model Description

- **Developed by:** [https://www.tii.ae](https://www.tii.ae);
- **Model type:** Causal decoder-only;
- **Language(s) (NLP):** English, German, Spanish, French (and limited capabilities in Italian, Portuguese, Polish, Dutch, Romanian, Czech, Swedish);
- **License:** [TII Falcon LLM License](https://huggingface.co/tiiuae/falcon-40b/blob/main/LICENSE.txt).

### Model Source

- **Paper:** *coming soon*.

## Uses

### Direct Use

Research on large language models; as a foundation for further specialization and finetuning for specific usecases (e.g., summarization, text generation, chatbot, etc.)

### Out-of-Scope Use

Production use without adequate assessment of risks and mitigation; any use cases which may be considered irresponsible or harmful. 

## Bias, Risks, and Limitations

Falcon-40B is trained mostly on English, German, Spanish, French, with limited capabilities also in in Italian, Portuguese, Polish, Dutch, Romanian, Czech, Swedish. It will not generalize appropriately to other languages. Furthermore, as it is trained on a large-scale corpora representative of the web, it will carry the stereotypes and biases commonly encountered online.

### Recommendations

We recommend users of Falcon-40B to consider finetuning it for the specific set of tasks of interest, and for guardrails and appropriate precautions to be taken for any production use.

## How to Get Started with the Model


```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-40b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

```

## Training Details

### Training Data

Falcon-40B was trained on 1,000B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb), a high-quality filtered and deduplicated web dataset which we enhanced with curated corpora. Significant components from our curated copora were inspired by The Pile ([Gao et al., 2020](https://arxiv.org/abs/2101.00027)). 

| **Data source**    | **Fraction** | **Tokens** | **Sources**                       |
|--------------------|--------------|------------|-----------------------------------|
| [RefinedWeb-English](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) | 75%          | 750B     | massive web crawl                 |
| RefinedWeb-Europe              | 7%           | 70B       | European massive zeb crawl                                   |
| Books  | 6%           | 60B        |                  |
| Conversations      | 5%           | 50B        | Reddit, StackOverflow, HackerNews |
| Code               | 5%           | 50B        |                                   |
| Technical          | 2%           | 20B        | arXiv, PubMed, UPSTO, etc.        |

RefinedWeb-Europe is made of the following languages:

| **Language** | **Fraction of multilingual data** | **Tokens** |
|--------------|-----------------------------------|------------|
| German       | 26%                               | 18B        |
| Spanish      | 24%                               | 17B        |
| French       | 23%                               | 16B        |
| _Italian_    | 7%                                | 5B         |
| _Portuguese_ | 4%                                | 3B         |
| _Polish_     | 4%                                | 3B         |
| _Dutch_      | 4%                                | 3B         |
| _Romanian_   | 3%                                | 2B         |
| _Czech_      | 3%                                | 2B         |
| _Swedish_    | 2%                                | 1B         |


The data was tokenized with the Falcon-[7B](https://huggingface.co/tiiuae/falcon-7b)/[40B](https://huggingface.co/tiiuae/falcon-40b) tokenizer.

### Training Procedure 

Falcon-40B was trained on 384 A100 40GB GPUs, using a 3D parallelism strategy (TP=8, PP=4, DP=12) combined with ZeRO.

#### Training Hyperparameters

| **Hyperparameter** | **Value**  | **Comment**                               |
|--------------------|------------|-------------------------------------------|
| Precision          | `bfloat16` |                                           |
| Optimizer          | AdamW      |                                           |
| Learning rate      | 1.85e-4       | 4B tokens warm-up, cosine decay to 1.85e-5 |
| Weight decay       | 1e-1       |                                           |
| Z-loss       | 1e-4       |                                           |
| Batch size         | 1152        | 100B tokens ramp-up                         |


#### Speeds, Sizes, Times

Training started in December 2022 and took two months. 


## Evaluation

*Paper coming soon.*

See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for early results.


## Technical Specifications 

### Model Architecture and Objective

Falcon-40B is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

The architecture is broadly adapted from the GPT-3 paper ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)), with the following differences:

* **Positionnal embeddings:** rotary ([Su et al., 2021](https://arxiv.org/abs/2104.09864));
* **Attention:** multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)) and FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135));
* **Decoder-block:** parallel attention/MLP with a two layer norms.

For multiquery, we are using an internal variant which uses independent key and values per tensor parallel degree.

| **Hyperparameter** | **Value** | **Comment**                            |
|--------------------|-----------|----------------------------------------|
| Layers             | 60        |                                        |
| `d_model`          | 8192      |                                        |
| `head_dim`         | 64        | Reduced to optimise for FlashAttention |
| Vocabulary         | 65024     |                                        |
| Sequence length    | 2048      |                                        |

### Compute Infrastructure

#### Hardware

Falcon-40B was trained on AWS SageMaker, on 384 A100 40GB GPUs in P4d instances. 

#### Software

Falcon-40B was trained a custom distributed training codebase, Gigatron. It uses a 3D parallelism approach combined with ZeRO and high-performance Triton kernels (FlashAttention, etc.)


## Citation

*Paper coming soon 😊.*

## License

Falcon-40B is made available under the [TII Falcon LLM License](https://huggingface.co/tiiuae/falcon-40b/blob/main/LICENSE.txt). Broadly speaking,
* You can freely use our models for research and/or personal purpose;
* You are allowed to share and build derivatives of these models, but you are required to give attribution and to share-alike with the same license;
* For commercial use, you are exempt from royalties payment if the attributable revenues are inferior to $1M/year, otherwise you should enter in a commercial agreement with TII.


## Contact
falconllm@tii.ae