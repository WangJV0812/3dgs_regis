# Mathematical formulas and derivations related to the project.

- [Mathematical formulas and derivations related to the project.](#mathematical-formulas-and-derivations-related-to-the-project)
  - [1. Gaussian Scene Similarity Transfromation](#1-gaussian-scene-similarity-transfromation)
  - [2. Log-space MLE in GMM](#2-log-space-mle-in-gmm)

## 1. Gaussian Scene Similarity Transfromation

The scale and origin of a gaussian scene are determined by the scale and origin of the training data. But in different application, the scale and origin are not always the same. Therefore, we need to apply a similarity transform to the gaussian scene to match the application requirements.

Assume a gaussian sphere in 3DGS Scene can be represented as:

$$
\mathcal G(\rm x \mid \rm \mu, \Sigma) = \rm C \exp \left(
    -\frac{1}{2} (\rm x - \rm \mu)^T \Sigma^{-1} (\rm x - \rm \mu)
\right)
$$

And we want to apply a similarity transform to the gaussian sphere. Actually, the transform is applied to the stochastic variable $\rm x$. The similarity transform can be represented as:

$$
\rm x' = s \rm R \rm x + \rm t
$$

So we have:

$$
x = \frac{1}{s} \rm R^T (\rm x' - \rm t)
$$

subsituting $\rm x$ into the gaussian spehere equation, we have:

$$
\begin{aligned}
    \mathcal G(\rm x \mid \rm \mu, \rm \Sigma)
    &= \rm C \exp \left(
        -\frac{1}{2} (\rm x - \rm \mu)^T \rm \Sigma^{-1} (\rm x - \rm \mu)
    \right)\\
    &= \rm C \exp \left(
        -\frac{1}{2} \left(
            \frac{1}{s} \rm R^T (\rm x' - \rm t) - \rm \mu
        \right)^T \rm \Sigma^{-1} \left(
            \frac{1}{s} \rm R^T (\rm x' - \rm t) - \rm \mu
        \right)
    \right)\\
    &= \rm C \exp \left(
        -\frac{1}{2} \left(
            \rm x' - (\rm t + s\rm R\rm \mu)
        \right)^T \frac{1}{s^2} \rm R \rm \Sigma^{-1} \rm R^T \left(
            \rm x' - (\rm t + s\rm R\rm \mu)
        \right)
    \right)\\
    &= \rm C \exp \left(
        -\frac{1}{2} \left(
            \rm x' - (\rm t + s\rm R\rm \mu)
        \right)^T \left(s^2 \rm R \rm \Sigma \rm R^T \right)^{-1}\left(
            \rm x' - (\rm t + s\rm R\rm \mu)
        \right)
    \right)\\
    &= \mathcal G(\rm x' \mid \rm \mu', \rm \Sigma')
\end{aligned}
$$

where:

$$
\begin{aligned}
    \rm \mu' &= \rm t + s\rm R\rm \mu\\
    \rm \Sigma' &= s^2 \rm R \rm \Sigma \rm R^T
\end{aligned}
$$

In 3dgs, we decompose the covariance matrix $\rm \Sigma$ into a rotation matrix $\rm R_{cov}$ and a scale matrix $\rm S_{cov}$:

$$
\rm \Sigma = \rm R_{cov} S_{cov} S_{cov} R_{cov}^T
$$

So the transformed covariance matrix can be represented as:

$$
\begin{aligned}
    R_{cov}' &= \rm R \rm R_{cov}\\
    S_{cov}' &= s S_{cov}
\end{aligned}
$$

In Vanilla 3DGS, we use sphere harmonics to represent the anisotropy of illuminance. Therefore, when applying similarity transform, we need to update the coefficients of sphere harmonics accordingly.

## 2. Log-space MLE in GMM

for any poiny $\rm x_i$ in the point cloud, we can find the top-k density gaussian episodes in the gaussian scene $\{\mathcal G\}_{j_1, j_2, \cdots j_k}$. Leveraging the locality of the gaussian scene, for point $\rm x_i$，the whole gaussian scene can be approximated as a mixture of the top-k density gaussian episodes. The GMM Model can be represented as:

$$
\text{GMM}_i = \sum_{j \in \{j_1, j_2, \cdots j_k\}} w_j \mathcal G_j(\rm x)
$$

In 3DGS scene, the weight $w_j$ is determined by the opacity of $j$-th gaussian episode $o_j$。Then the likelihood of point $\rm x_i$ can be represented as:

$$
\log p(\rm x_i) = \log \sum_{j \in \{j_1, j_2, \cdots j_k\}} w_j \mathcal G_j(\rm x)
$$

In registration task (means thart using mle to optimize a Rts between 3dgs scene and pointcloud), we do not needed to estimate the parameters in 3dgs scens, se the normalized of gmm model is not necessary actually.

