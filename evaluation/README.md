# Evaluation

Normally, evaluation is done following the training. But if you want to do evaluation manually using the output files of a subsequent test run, you can do this using _evaluation.py_ and the following command:

```bash
> python evaluate.py --all=True \
    --pred=/absolute/path/to/test_predictions_XX.txt \
    --sys=/absolute/path/to/test_targets_0.txt \
```

The scripts outputs the evaluation results directly to the console.

If you want to use PARENT, you can do this by using the _calc_parent.py_ script. The script does not need any command-line arguments. Please set the constants directly in the script.

If you want to calculate the BLEURT score (in the table-to-text scenario), please use this command:

```bash
> python -m bleurt.score_files \
    -candidate_file=/absolute/path/to/test_predictions_XX.txt \
    -reference_file=/absolute/path/to/test_targets_0.txt \
    -bleurt_checkpoint=/absolute/path/to/downloaded/and/unziped/bleurt/model
```

