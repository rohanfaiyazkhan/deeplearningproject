# Reproduction of "Facial recognition technology can expose political orientation from naturalistic facial images"

This project is aimed at reproduction and analysis of the methods described in the paper “Facial recognition technology can expose political orientation from naturalistic facial images” [\[1\]](https://www.nature.com/articles/s41598-020-79310-1).

## Requirements

To install requirements, run:

```bash
pip install -r requirements.txt
```

Datasets can be downloaded from the link specified in the paper. [https://osf.io/c58d3/](https://osf.io/c58d3/)

TODO: Write script for processing RData files for consumption by pandas.

## Code

The relevant code is as follows:

- A copy of the original R-script provided by the authors in `code.R`.
- Lasso-regression, the first method specified in the paper, is approximated in python in `py-reproduction.ipynb`
- A neural network approach (mentioned in author's paper but not provided by the author) is implemented in `simple-nn.ipynb`

TODO: Extract code for training and evaluation and specify how to run the scripts.

## Reference

[\[1\] M. Kosinski, “Facial recognition technology can expose political orientation from naturalistic facial images,” Scientific Reports, vol. 11, no. 1, Art. no. 1, Jan. 2021, doi: 10.1038/s41598-020-79310-1.](https://www.nature.com/articles/s41598-020-79310-1)


