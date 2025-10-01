# AHaBench

This is the official repository for our paper:

> **"Being Kind Isn’t Always Being Safe: Diagnosing Affective Hallucination in LLMs"**

## 📊 Dataset

You can download the **AHaBench** and **AHaPairs** datasets from [Hugging Face](https://huggingface.co/datasets/anonymous9268/AHaBench).

## 🚀 How to Use

### 🛠️ Dependencies

- **Python 3.10**
- **Python packages**  
  Install the required packages with:
  ```bash
  pip install -r requirements.txt
  ```
- **For DPO Training**
    Requires at least one A100 GPU node.

- **For Evaluation Metrics**
    Uses the OpenAI API, only CPU is needed.

### 🏋️‍♂️ DPO Training
Run the following command to train with Direct Preference Optimization (DPO):
```sh
CUDA_VISIBLE_DEVICES=${device_nums} python dpo.py \
  --input_path data/train.json \  # change AHaPairs.json to train.json if needed
  --base_model ${model_name} \
  --output_path ./dpo_output \
  --hug_token ${huggingface_token}
```
> **Note:** 
> Replace device_nums, model_name, and huggingface_token with your actual values.


### 🧑‍🏫 SFT Training
For Supervised Fine-Tuning (SFT), use the following command:
```sh
CUDA_VISIBLE_DEVICES=${device_nums} python sft.py \
  --base_model ${model_name} \
  --output_path ./sft_output \
  --hug_token ${huggingface_token}
```

### 📝 Evaluation
To evaluate using the metric prompts, run:
```sh
python metric_prompt.py --api-key ${openai_api_key}
```
> **Note:** 
> You'll need an OpenAI API key for evaluation.

## 📄 Citation
If you use this repository or dataset in your research, please cite our paper (link to be added after publication).
