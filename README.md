# FarconVAE: FAir Representation via distributional CONtrastive VAE
This repository is the official PyTorch Implementation of [Learning Fair Representation via Distributional Contrastive Disentanglement](https://arxiv.org/abs/2206.08743) (KDD 2022)



## Abstract
<img align="middle" width="100%" src="FarconVAE.png">

Learning fair representation is crucial for achieving fairness or debiasing sensitive information. Most existing works rely on adversarial representation learning to inject some invariance into representation. However, adversarial learning methods are known to suffer from relatively unstable training, and this might harm the balance between fairness and predictiveness of representation. We propose a new approach, learning _FAir Representation via distributional CONtrastive Variational AutoEncoder (**FarconVAE**)_, which induces the latent space to be disentangled into sensitive and nonsensitive parts. We first construct the pair of observations with different sensitive attributes but with the same labels. Then, FarconVAE enforces each non-sensitive latent to be closer, while sensitive latents to be far from each other and also far from the non-sensitive latent by contrasting their distributions. We provide a new type of contrastive loss motivated by Gaussian and Student-t kernels for distributional contrastive learning with theoretical analysis. Besides, we adopt a new swap-reconstruction loss to boost the disentanglement further. FarconVAE shows superior performance on fairness, pretrained model debiasing, and domain generalization tasks from various modalities, including tabular, image, and text.

## Contribution
* We propose a novel framework FarconVAE that learns disentangled invariant representation with contrastive loss to achieve algorithmic fairness and domain generalization.
* We provide new distributional contrastive losses for disentanglement motivated by the Gaussian and Student-t kernels.
* The proposed method is theoretically analyzed and empirically demonstrated on a broad range of data types (tabular, image, and text) and tasks, including fairness and domain generalization


## Prepare your environment
```
conda create -n farcon python=3.8
conda activate farcon 
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install datetime
git clone https://github.com/changdaeoh/FarconVAE.git
cd FarconVAE
```

## Run
To reproduce fair classification task, now you can run FarconVAE with `run_xxx.sh`:
```
sh run_yaleb.sh
```

## Citation

```
@article{oh2022learning,
  title={Learning Fair Representation via Distributional Contrastive Disentanglement},
  author={Oh, Changdae and Won, Heeji and So, Junhyuk and Kim, Taero and Kim, Yewon and Choi, Hosik and Song, Kyungwoo},
  journal={arXiv preprint arXiv:2206.08743},
  year={2022}
}
```
