# Measurements

When running `python ./harness/run_submission.py <some-variant>`, it will generate measurement files in a sub-directory under this directory.
Specifically, the sub-directories that it uses are `single`, `small`, `medium` and `large` for the ml-inference variants.

If it is run with argument `--num_runs <n>` it will generate `<n>` measurements files called `results-1.json`, ..., `results-<n>.json`, all in the same sub-directory.

Before submitting your implementation, run the `run_submission.py` script with argument `--num_runs 3` for each variant of the workload that you want to submit. Then commit all these results file to your fork, the average of these three runs will be the numbers reported for your submission.

## Results for the reference implementation

For the reference implementation we only produced measurements for single, and small instances. You can find these measurements in the sub-directories here, they were generated in July 2026.
