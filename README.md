# RAG for documents: containing text paired with metadata

## how to run

### Step 1: environment setup

```bash
conda env create -f mmrag_env.yaml

conda activate mmrag
```

if encounting with cuda version problem, you can alse try this:
```bash
cd Multimodal-Rag

pip install torch torchvision torchaudio  # install your correct torch verion

pip install -r requirements.txt
```
### Step 2: configurate

- fill in environment variables  scripts\run.sh

### Step 3:

```bash
bash script/run.sh
```


To change parameters, please change in `configs.yml`.
