# FPBench: A Comprehensive Benchmark of Multimodal Large Language Models for Fingerprint Analysis
<i>Ekta Gavas, Sudipta Banerjee, Chinmay Hegde, Nasir Memon</i>

[![arXiv](https://img.shields.io/badge/arXiv-2405.09882-red)](https://arxiv.org/abs/2512.18073)

Official implementation of paper "FPBench: A Comprehensive Benchmark of Multimodal Large Language Models for Fingerprint Analysis".


## Abstract

Multimodal LLMs (MLLMs) are capable of performing complex data analysis, visual question answering, generation, and reasoning tasks. However, their ability to analyze biometric data is relatively underexplored. In this work, we investigate the effectiveness of MLLMs in understanding fine structural and textural details present in fingerprint images. To this end, we design a comprehensive benchmark, FPBench, to evaluate 20 MLLMs (open-source and proprietary models) across 7 real and synthetic datasets on a suite of 8 biometric and forensic tasks (e.g., pattern analysis, fingerprint verification, real versus synthetic classification, etc.) using zero-shot and chain-of-thought prompting strategies. We further fine-tune vision and language encoders on a subset of open-source MLLMs to demonstrate domain adaptation. FPBench is a novel benchmark designed as a first step towards developing foundation models in fingerprints. Our findings indicate fine-tuning of vision and language encoders improves the performance by 7%-39%. 


Our key contributions are:
- We introduce FPBench, a first-of-its-kind evaluation benchmark for fingerprint understanding using MLLMs. We evaluate 18 open-source and 2 proprietary MLLM models on a suite of biometric and forensic tasks using real and synthetic datasets.
- We investigate zero-shot and chain-of-thought (CoT) prompt design strategies to guide the MLLMs to perform complex feature analysis for verification and classification tasks using structured reasoning.  
- We further investigate the impact of fine-tuning of vision and language encoders for successful domain adaptation, and compare our findings with state-of-the-art baselines in fingerprint analysis.



<img src="/images/task_cat.png" height="500"/>

![](/images/results.png)
![](/images/ft_results.png)


## Data preparation
Obtain and download fingerprint datasets from original sources:
- [NIST SD301a](https://www.nist.gov/itl/iad/btg/nist-special-database-301)
- [NIST SD302d](https://www.nist.gov/itl/iad/image-group/nist-special-database-302)
- [FVC2000](http://bias.csr.unibo.it/fvc2000/databases.asp)
- [FVC2002](http://bias.csr.unibo.it/fvc2002/databases.asp)
- [FVC2004](http://bias.csr.unibo.it/fvc2004/databases.asp)
- [GenPrint](https://biometrics.cse.msu.edu/Publications/Databases/MSU_GenPrint/)
- [Anguli 10K](https://dsl.cds.iisc.ac.in/projects/Anguli/#download)

### Minutiae extraction, orientation estimation and matching ground-truths
We setup NBIS software from NIST following instructions from their [repo](https://github.com/lessandro/nbis) and utilise MINDTCT, BOZORTH3 to obtain ground-truths for various tasks and also establish verification baseline.

Additionally, transformer-based matching baseline is established by following the steps from this [paper](https://github.com/saraansh1999/global-plus-local-fp-transformer). 

The scripts to generate minutiae and orientation overlays are provided in `FPBench/misc`.

## Setup
Download and setup VLMEvalKit from https://github.com/open-compass/VLMEvalKit.git

To setup FPBench, follow the instructions below:
```shell
git clone https://github.com/Ektagavas/FPBench.git
mv FPBench/* VLMEvalKit/

# Here task name can be one of the eight tasks viz. 'pattern','matching','minutiae','sensor','orientation','realvssyn','ace','tools', or 'all' to run evaluation on all tasks
python VLMEValKit/evaluate.py --model <chosen_model> --cat <task_name>

# OR To run chain-of-thought evaluation
python VLMEValKit/evaluate_cots.py --model <chosen_model> --cat <task_name>

# Aggregate results by extracting correct option from prediction
# and get performance analysis on different categories and sub-categories in the benchmark.
python VLMEValKit/aggregate_results.py \
  --model <chosen_model>
  --results_dir <path_to_results_dir>

# The results are stored at <path_to_results_dir>/<chosen_model>/results.txt

# Save results for all models and tasks in model_accuracies.csv
python save_results.py

# OR To save results for COT evaluation
python save_results_cot.py

```

Note: To run GPT-5, Gemini Pro 2.5 and Qwen3, run the corresponding scripts provided in `FPBench`. For GPT and Gemini, user needs to obtain the API keys and replace the values in 'YOUR_API_KEY_HERE' in the respective files.

## Chain-of-thought setting
Use "xxx_cot.py" files for chain-of-thought evaluation for Qwen3, Gemini and GPT, which in turn uses "xxx_cot.json" benchmark files for evaluation.

## Finetuning
We finetuned Gemma3-12B, Qwen3-8B and Qwen3-32B on seven tasks in FPBench. The finetuning question set and scripts are placed in `finetuning`.
```shell
# Run this script and change model Qwen-8B or Qwen-32B. Use either vision or llm options to train corresponding layers or both.
# Fine-tuned LORA adapters are stored in folder path given with output option.
# Change input dataset paths in file based on selected task.

python ft_qwen3.py --model Qwen/Qwen3-VL-8B-Instruct --vision --llm --output <output_path>

# OR to finetune Gemma3-12B
python ft_gemma3.py --vision --llm --output <output_path>

```
Finetuned checkpoints can be accessed from [here](https://drive.google.com/file/d/1uFQlqDLzwqAPvbbMBNWarOkGhAR-C32Z/view?usp=drive_link).

## Demo
To run demo, we provided sample images and benchmarks in `demo` folder. Run the following code to run demo on pattern task with sample images. Results will be saved in `demo/results/<chosen_model>/`.
```shell
cd demo
python evaluate.py --model <model_name> --cat pattern
```

## Responsible Usage
This code is provided for academic and research purposes in connection with the paper "FPBench: A Comprehensive Benchmark of Multimodal Large Language Models for Fingerprint Analysis". The authors have not systematically evaluated potential data leakage or memorization issues, including those that may arise from fine-tuned models or downstream applications. Commercial use is not intended or supported.
Please cite the paper when using this code. 
```bibtex
@article{gavas2025fpbench,
  title={FPBench: A Comprehensive Benchmark of Multimodal Large Language Models for Fingerprint Analysis},
  author={Gavas, Ekta Balkrishna and Banerjee, Sudipta and Hegde, Chinmay and Memon, Nasir},
  journal={arXiv preprint arXiv:2512.18073},
  year={2025}
}
```

## Acknowledgments
Our code structure is based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)


## Contact
If you have any questions, please contact at eg4131@nyu.edu
