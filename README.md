# Fairness in Ranking under Disparate Uncertainty
This repository contains the official implementation of the paper [[Fairness in Ranking under Disparate Uncertainty]](https://arxiv.org/pdf/2309.01610). 

| **[Environment Setup](##Environment)**
| **[Example](##Example)**
| **[Citation](##Citation)**

Ranking is a ubiquitous method for focusing the attention of human evaluators on a manageable subset of options. Its use as part of human decision-making processes ranges from surfacing potentially relevant products on an e-commerce site to prioritizing college applications for human review. While ranking can make human evaluation more effective by focusing attention on the most promising options, we argue that it can introduce unfairness if the uncertainty of the underlying relevance model differs between groups of options. Unfortunately, such disparity in uncertainty appears widespread, often to the detriment of minority groups for which relevance estimates can have higher uncertainty due to a lack of data or appropriate features. To address this fairness issue, we propose Equal-Opportunity Ranking (EOR) as a new fairness criterion for ranking and show that it corresponds to a group-wise fair lottery among the relevant options even in the presence of disparate uncertainty. EOR optimizes for an even cost burden on all groups, unlike the conventional Probability Ranking Principle, and is fundamentally different from existing notions of fairness in rankings, such as demographic parity and proportional Rooney rule constraints that are motivated by proportional representation relative to group size. To make EOR ranking practical, we present an efficient algorithm for computing it in time O(nlog(n)) and prove its close approximation guarantee to the globally optimal solution. In a comprehensive empirical evaluation on synthetic data, a US Census dataset, and a real-world audit of Amazon search queries, we find that the algorithm reliably guarantees EOR fairness while providing effective rankings.

<!-- <img src="./posterior.png" alt="Illustration of Disparate Uncertainty between two groups" width="300"/> -->

## Environment
Python 3.7.12
Set up and activate the Python environment by executing

- Create virtual Enviroment with virtualenv, conda or ...

- Install required libraries

```shell
pip install -r requirements.txt
```

## Example
<!-- Notebook  "Duality Thm1" for evaluating the theorem 6.1 example of cost optimality gap.

Notebook  "Markup_Amazon_Analysis" for analysis of Amazon search queries experiment.

Notebook  "USCensus_experiment" and "Census_plot" for experiment and plotting of USCensus dataset. -->

Notebook  "Example" demonstrates a synthetic example with all baselines for a given calibrated expected relevance estimates $P(r_i|D)$. -->

## Citation

```
@misc{rastogi2023fairness,
    title={Fairness in Ranking under Disparate Uncertainty},
    author={Richa Rastogi and Thorsten Joachims},
    year={2023},
    eprint={2309.01610},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

