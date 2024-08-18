# Llama3 Fine Tuning for ABAP Code Generation

This repository contains the work done as part of my bachelor's thesis, where I fine-tuned the Llama3.1:8B language model for generating ABAP code using Unsloth.

The models and datasets are available on my [HuggingFace profile](https://huggingface.co/timkoehne)

# Training Data
 - DS1 contains almost 30,000 ABAP files of reports, classes and function-modules extracted from [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)

- DS2 contains 1,000 examples from [Code-Alpaca](https://github.com/sahil280114/codealpaca) which have been translated into ABAP using a llama3

- DS3 was the only publicly available non trivial ABAP [dataset on HuggingFace](https://huggingface.co/datasets/smjain/abap) at the time


# Result
After fine-tuning 10 models using 4-Bit QLoRA based on three different dataset there was no significant improvement compared to Meta's Llama3.1:8B-Instruct model.

There is just not enough high-quality ABAP data available to train a useful model. 

In the future maybe synthetic data can be used to try to improve these results.

