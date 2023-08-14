# Logistic-Normal Likelihoods for Heteroscedastic Label Noise

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2304.02849-green)](https://arxiv.org/abs/2304.02849)

</div>


This is the official code repository for the TMLR 2023 paper [Logistic-Normal Likelihoods for Heteroscedastic Label Noise](https://openreview.net/forum?id=7wA65zL3B3).


## Abstract
A natural way of estimating heteroscedastic label noise in regression is to model the observed (potentially noisy) target as a sample from a normal distribution, whose parameters can be learned by minimizing the negative log-likelihood. This formulation has desirable loss attenuation properties, as it reduces the contribution of high-error examples. Intuitively, this behavior can improve robustness against label noise by reducing overfitting. We propose an extension of this simple and probabilistic approach to classification that has the same desirable loss attenuation properties. Furthermore, we discuss and address some practical challenges of this extension. We evaluate the effectiveness of the method by measuring its robustness against label noise in classification. We perform enlightening experiments exploring the inner workings of the method, including sensitivity to hyperparameters, ablation studies, and other insightful analyses.

## Environment Setup
The code is based on, and requires, the [Uncertainty Baselines repository](https://github.com/google/uncertainty-baselines), please follow the installation instructions there. 

Our experiments were run with TensorFlow 2.6.0, TensorFlow Probability 0.14.1 and Uncertainty Baselines 0.0.7.

## Running Experiments
For example, to run the Logistic-Normal (LN) method on CIFAR-10 with 40% asymmetric noise, run the following
```bash
python baselines/cifar/ln.py --data_dir=/path/to/data/ \
                             --output_dir=/path/to/output_dir/ \
                             --dataset cifar10 \
                             --noisy_labels \
                             --severity 0.4 \
                             --corruption_type asym \
                             --temperature 0.5 \
                             --min_scale 0.5
```
or LN on Clothing1M
```bash
python baselines/clothing1m/ln.py --data_dir=/path/to/data/ \
                                  --output_dir=/path/to/output_dir/ \
                                  --temperature 1.0 \
                                  --min_scale 1.0
```


where `temperature` and `min_scale` corresponds to the hyperparameters $\tau$ and $\lambda$ in the paper, respectively. Please see Tables 5 and 6 in the Appendix of the paper for the hyperparameters used in other settings.
