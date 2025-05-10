# QianKun

This repository contains QianKun's implementation code and experimental logs. You can reproduce the experimental results from the paper by running the source code in the `src` directory or directly generate plots using the provided experimental logs.

## Contents

- `src/`: Source code implementation
  - `llm_model/`: LLM prediction model
  - `Design/`: QianKun as an ABR design component
  - `Testing/`: QianKun for automated testing
  - `Optimization/`: QianKun for automated optimization
- `logs/`: Experimental logs (currently includes data for Figures 6, 7, 14 and Tables 2, 3)

## Getting Started

### Prerequisites

- Python 3.11
- Meta-Llama-3.1-8B-Instruct model

### Installation

1. Clone this repository

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download Meta-Llama-3.1-8B-Instruct model and configure the path in `src/llm_model/models/configs.py` (update the `llm_ckp_dir` parameter).

4. Place the provided fine-tuned model checkpoint (`checkpoint.pth`) in the `src/llm_model/models/` directory.

## Usage

### Design Phase

To reproduce results from the algorithm design section:
```
cd src/Design
python mpc_llm.py [dataset_name]
```

### Testing Phase

To obtain results from the algorithm testing section:
```
cd src/Testing
python agent.py [dataset_name]
```

You can choose to end testing automatically using `last_evaluation` or specify a fixed dataset proportion for testing.

### Optimization Phase

To optimize the framework on a specific dataset:
```
cd src/Optimization
python agent.py [dataset_name]
```

We provide optimized parameters in `log/Optimization/bayes_rules`. To reproduce the results for Figure 14 and Table 3, copy bayes_rules to src/Optimization:
```
cd src/Optimization
python bayes_mpc.py [dataset_name]
```

## Experimental Logs

Due to size constraints, the current logs only include the main experimental results for algorithm design and optimization:
- Figures 6, 7, 14
- Tables 2, 3

For other experimental results, please run the corresponding code in the `src` directory.