# Decision Entropy Experiments

## Setup

**Install dependencies**

   All required Python packages are listed in `requirements.txt`.  
   You can install them into a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate      # on Linux/macOS
   venv\Scripts\activate         # on Windows

   pip install -r requirements.txt
   ```

---

## Running Experiments Locally

To run an experiment, use the command:

```bash
python decision_entropy_full_experiment     --model <model_name>     --dataset <dataset_name>     --hp-opt     --path <output_path>
```

### Arguments
- `--model`: Name of the model to use  
- `--dataset`: Name of the dataset  
- `--hp-opt`: Enables hyperparameter optimization  
- `--path`: Path where results will be stored

## Running on Mogon

### 1. Single Job Script

To submit a single experiment via SLURM, use:

```bash
sbatch job_script.sh
```

Before running, **edit** `job_script.sh`:
- Adjust experiment flags (`--model`, `--dataset`, etc.)
- Set the correct **path to your virtual environment** for activation

---

### 2. Job Array Script

To run **all model–dataset combinations** as a job array, use:

```bash
sbatch --array=0-11 job_script_array.sh
```

Before running, ensure:
- The array indices (`0-11`) match the number of combinations defined in your script.
- The correct **virtual environment path** is set inside `job_script_array.sh`.

