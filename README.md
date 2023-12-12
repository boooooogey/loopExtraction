# loopExtraction
Explanation for the terminal inputs:
- `--HiC`: To pass the path to the HiC matrix. Data is assumed to be in NumPy format. example: HiC.npy.
- `--bin-size`: Size of a bin in HiC matrix provided. Not used right now other than writing outputs to a file with start and end loci.
- `--chromosome-name`: Name of the chromosome the HiC matrix belongs to. Used for the output file.
- `--model`: Name of the model to be fitted to the data. The name should correspond to a file under `model` folder. Within the file there should be a function named `model` that returns the necessary functions to fit the data. Check `one_lambda_stickiness_l2_norm.py` file under model folder for an example.
- `--model-output`: Path to write the output to. The output file is tab separated file with first three columns are chromosome names, start and end positions, and the rest is the corresponding parameters.
- `--window-size`: Size of the patch to be taken from the diagonal of the HiC matrix. The default is 400.
- `--overlap`: Overlap between the consecutive patches. The default is 200.
- `--transformation`: The transformation to be applied on the patches. It should correspond to a function name in `transformations.py`. The default is identity. There is currently one transformation in `transformation.py` which is `log_transform`. It is used for MicroHiC.

Example run:

`python fit_model.py --HiC data/4DNFI9GMP2J8_chr10_res_8000_unbalanced.npy --bin-size 8000 --chromosome-name chr10 --model one_lambda_stickiness_l2_norm --model-output local/4DNFI9GMP2J8_chr10_res_8000_unbalanced.tsv --transformation log_transform`
