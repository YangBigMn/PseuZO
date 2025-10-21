# PseuZO: Pseudo-Zeroth-Order Algorithm for Training Deep Neural Networks(NeurIPS2025)
This project includes code adapted from:

-[MeZO] https://github.com/princeton-nlp/MeZO, licensed under the MIT License.
## Installation
```
conda create -n PseuZO python=3.9.7
conda activate PseuZO
pip install -r requirements.txt
```

## Usage
Change the trainer from zo to pzo(PseuZO).
```bash
# PseuZO (full-parameter, prefix-tuning, and LoRA)
CUDA_VISIBLE_DEVICES=0 TRAINER=pzo MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash mezo.sh
CUDA_VISIBLE_DEVICES=0 TRAINER=pzo MODEL=facebook/opt-1.3b TASK=SST2 MODE=prefix LR=1e-3 EPS=1e-1 bash mezo.sh
CUDA_VISIBLE_DEVICES=0 TRAINER=pzo MODEL=facebook/opt-1.3b TASK=SST2 MODE=lora LR=5e-5 EPS=1e-2 bash mezo.sh
```
## Tips
Since we need the gradient of the last layer, the forward propagation process will be somewhat different from the usual forward propagation. If you are interested in this, you can check the implementation of forward propagation in $\textit{util.py}$. Some directory locations may cause errors. Please check whether the directory locations in your related hyperparameters and code correspond correctly.
