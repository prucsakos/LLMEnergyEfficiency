vLLM quantization methods:
['awq', 'deepspeedfp', 'tpu_int8', 'fp8', 'ptpc_fp8', 'fbgemm_fp8', 'modelopt', 'modelopt_fp4', 'bitblas', 'gguf', 'gptq_marlin_24', 'gptq_marlin', 'gptq_bitblas', 'awq_marlin', 'gptq', 'compressed-tensors', 'bitsandbytes', 'hqq', 'experts_int8', 'ipex', 'quark', 'moe_wna16', 'torchao', 'auto-round', 'rtn', 'inc', 'mxfp4', 'petit_nvfp4']. [type=value_error, input_value=ArgsKwargs((), {'model': ...rocessor_plugin': None}), input_type=ArgsKwargs]

# Plan
- Calib: default deepspeedfp - if bnb 4bit in name then no additional quant
- Bench: default bitsandbytes - if bnb 4bit in name then no additional qunat


TEST 1 - deepspeed quantization:
Use HF+BnB to load the models as quantized 4bit
Wrap THAT into deepspeed inference negine.  



----
10.06.
Sometimes small models start to repeat themself. Why? Temperature settings? 

I did set judge and direct answer to temperature 0 and top_p 1

## 10.07. Conclustions - Possible Improvements
#### Models do not always utilize the full thinking budget. They stop thinking when they arrived to a final answer - even if it is wrong.
- This is for simple CoT. It can be improved to double check it's final answer.
- Try other reasoning methods like multi-pass / self-consistency. Compare their performance.
#### gpqa: multiple choice task: self-judge declined choosing the right option, but without the text. Ex: "A)" Instead of "A) the answer"
- Possibly i need to modify the system prompt for the judge llm. 

### Qwen 4b Thinking reaches published performance on AIME2025 at 20k thinking budget!
<br><br>

# Problem> Models are unaware of thinking budget.
There are two typical cases: 1. The model finishes thinking, concludes to a answer, then succeeds the task.
2. The thinking is terminated half-ways because thinking buged was exceeded. Therefore the thinking process lacks of conclusions. The model fails to answer the question.

Without the models being aware of their thinking budget, the measurment feels fake. Poor accuracy is not caused by model errors but measurement errors.
-> Publications about budget aware thinking models exist.

### Possible solutions
- Budget Aware Prompting
- Dynamic Budget Communication (nematron does this?)

# Nemotron avoids thinking or buged - aime2025
 - First pass: ""
 - Second pass: \<wrong answer\>

<br><br><br><br>
--------
--------
--------

# 10/22
## Key metrics that I capture
#### Resources
<i>Total and Per Datapoint Avg might be missleading as the models can fall into curse of repetition capping out the token budget -> corrupting the metrics regarding resources (Energy, Num Tokens Gen, FLOP Estimation)</i>
- Joules consumed on benchmark
- Total Number of Tokens Generated | Totel num of tokens on correct predictions only
- FLOP estimation by extrapolation model - Total | On Corrects only

## Meaningful plots
- Pareto frontier of models (Min resource, Max accuracy)
- Trend line of Models by release date and Accuracy - With Generated Tokens on Correct Prediction to highlight reasoning quality
- Leaderboard of models on Efficiency Score: Accuracy/Resource | Accuracy/(Resource on correct) groupped by accuracy bins
- Model quality changes on Various Types of Benchmark - Math, GeneralQA, Etc.. - This can be a table?  

## Datasets
- Math: AIME25, HMMT_FEB25 30-30 thinking question
- Easy Math: gsm8k - or a subset of it. 
- Mixed Subhects: GPTQ-Diamond - Physics, Biology, Chemistry - Multi Choice
- Any other?

## Flop Calibration 
- Linear regression with second degree polinomial features - sklearn
- Calib. Dataset is generated with random tokens by given model's Tokenizer. 