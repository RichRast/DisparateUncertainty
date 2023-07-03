# Fair Ranking under Disparate Uncertainty

| **[Installation](#installation)**
| **[Examples](#examples)**
| **[Citation](#citation)**

Ranking is a ubiquitous method for focusing the attention of human evaluators on a manageable subset of options. Its use ranges from surfacing potentially relevant products on an e-commerce site to prioritizing college applications for human review. While ranking can make human evaluation far more effective by focusing attention on the most promising options, we argue that it can introduce unfairness if the epistemic uncertainty differs between groups of options. Unfortunately, such disparity in epistemic uncertainty appears widespread, since the relevance estimates for minority groups tend to have higher uncertainty due to a lack of data or appropriate features. To overcome this fairness issue, we propose Equal-Opportunity Ranking (EOR) as a new fairness criterion for ranking that provably corrects for the disparity in epistemic uncertainty between groups. Furthermore, we present a practical algorithm for computing EOR rankings in time $O(n \log(n))$ and prove a close approximation guarantee to the intractable integer programming solution. We evaluated the efficacy of our algorithm with empirical experiments on synthetic data and a real-world case study of Amazon search queries.

<img src="posterior.pdf" alt="Disparate Uncertainty between two groups">

## Acknowledgements



## Installation

Set up and activate the Python environment by executing

```
conda env create -f environment.yml
conda activate ranking_uncertainty
```

SLURM system can be used to run jobs. An example script for submitting SLURM job is given in ```./scripts/combined_sbatch.sub```.
In the scripts folder, customize the script ```init_env.sh``` for your environment and path. This path is then referenced in ```./scripts/combined_sbatch.sub``` .


## Examples


## Citation
If you find this repo useful for your research, please consider citing our paper:
```

```

## Feedback
For any questions/feedback regarding this repo, please contact [here](rr568@cornell.edu)