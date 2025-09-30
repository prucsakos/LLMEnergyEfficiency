## Project Content
- Benchmarking framework for LLMs with reasoning.
- Models run in VLLM
- A few math benchmarks
### Reasoning
- CoT reasoning and SC.
- Self-Consistency runs K CoT passes, each with different temperature and top-k parameters for divergent reasoning paths.
- Baseline prompts for CoT/Direct Answer/Judge/Self-Consistency etc...
- Baseline prompts contain XML like elements.
### Reasoning - Thinking budget
- Thinking budget as a parameter
- Thinking is terminated without warning if reasoning goes over budget.
### Logging
- W&B Experiment tracking woth detailful parameters
- Plots of performance
### Flop estimation
- Flop usage is estimated by extrapolation. 
- Models generate a calibration set with DeepSpeed (measures flop) - regressor is fit on this set.
### Evaluation
- Self Evaluation - Judge LLM
- OpenAI gpt-5-mini | self-eval on fallback. 

### Models
- Phi, Llama, Nemotron, Qwen, Deepseek, Gemma
- Above 40B parameters: quantized versions with cpu offloading
## Dataset
#### HMMT Feb 2025 - Harvard-MIT Math Tournament
- Final Answer competition of 30 questions. 
- Field: Math
#### AIME 2025 - American Invitational Mathematics Examination
- Final Answer competition of 30 questions: [AIME-25-I](https://artofproblemsolving.com/wiki/index.php/2025_AIME_I), [AIME-25-II](https://artofproblemsolving.com/wiki/index.php/2025_AIME_II)
- Field: Math
#### GPQA Main - Graduate-Level Google-Proof Q&A
- Multi Choice / Final Answer competition of 443 questions.
- Field: Physics, Biology, Chemistry

## Improvements
- Use the specified chat template for each model instead of universal prompts. 
- Better prompts?