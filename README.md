# Fair Ranking under Disparate Uncertainty

| **[Installation](#installation)**
| **[Examples](#examples)**
| **[Citation](#citation)**

Ranking is a ubiquitous method for focusing the attention of human evaluators on a manageable subset of options. Its use ranges from surfacing potentially relevant products on an e-commerce site to prioritizing college applications for human review. While ranking can make human evaluation far more effective by focusing attention on the most promising options, we argue that it can introduce unfairness if the uncertainty of the underlying relevance model differs between groups of options. Unfortunately, such disparity in uncertainty appears widespread, since the relevance estimates for minority groups tend to have higher uncertainty due to a lack of data or appropriate features. To overcome this fairness issue, we propose Equal-Opportunity Ranking (EOR) as a new fairness criterion for ranking that provably corrects for the disparity in uncertainty between groups. Furthermore, we present a practical algorithm for computing EOR rankings in time $O(n\log{n})$ and prove its close approximation guarantee to the globally optimal solution. In a comprehensive empirical evaluation on synthetic data, a US Census dataset, and a real-world case study of Amazon search queries, we find that the algorithm reliably guarantees EOR fairness while providing effective rankings.

<!-- ![Illustration of Disparate Uncertainty between two groups](./posterior.png)  -->
<img src="./posterior.png" alt="Illustration of Disparate Uncertainty between two groups" width="300"/>
<!-- <embed src="./posterior.pdf" > -->

<!-- ## Acknowledgements -->



## Installation

Set up and activate the Python environment by executing

```
conda env create -f environment.yml
conda activate ranking_uncertainty
```

<!-- SLURM system can be used to run jobs. An example script for submitting SLURM job is given in ```./scripts/combined_sbatch.sub```.
In the scripts folder, customize the script ```init_env.sh``` for your environment and path. This path is then referenced in ```./scripts/combined_sbatch.sub``` . -->


## Examples
Notebook  "Duality Thm1" for evaluating the theorem 6.1 example of cost optimality gap.

Notebook  "Markup_Amazon_Analysis" for analysis of Amazon search queries experiment.

Notebook  "USCensus_experiment" and "USCensus_plot" for experiment and plotting of USCensus dataset.

Notebook  "FairRanking_DisparateUncertainty" can be used to run synthetic examples for given $G$ groups and specific expected relevance estimates $P(r_i|D)$.

## Citation
If you find this repo useful for your research, please consider citing our paper:
```
@misc{rastogi2023fair,
      title={Fair Ranking under Disparate Uncertainty}, 
      author={Richa Rastogi and Thorsten Joachims},
      year={2023},
      eprint={2309.01610},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Feedback
For any questions/feedback regarding this repo, please contact [here](rr568@cornell.edu)