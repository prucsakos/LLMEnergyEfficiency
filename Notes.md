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

