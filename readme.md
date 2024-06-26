Unofficial `foreach` implementation of [Adan](https://github.com/sail-sg/Adan) (Adaptive Nesterov Momentum) with [Schedule-Free](https://github.com/facebookresearch/schedule_free). 

> [!NOTE]
> To use this optimizer `optimizer.train()` and `optimizer.eval()` must be called at the same place where `model.train()` and `model.eval()` are called. The optimizer should also be placed in eval mode when storing checkpoints.

Code developed on python 3.12 and pytorch =>2.3

## experiments

In order to test the potential of Adan Schedule-Free, a small experiment was conducted using SISR (single-image super-resolution) under the same, bit-wise [deterministic](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html), setting. The [SPAN](https://github.com/hongyuanyu/SPAN) network was trained up to 180k iters, reducing [charbonnier](https://github.com/muslll/neosr/blob/master/neosr/losses/basic_loss.py#L133) loss. Both optimizers used the same learning rate of `2.5e-3`. For AdamW Schedule-Free, betas `[0.9, 0.99]` were used, no decay. For Adan Schedule-Free, betas `[0.98, 0.92, 0.987]` and decay `0.02` were used. Results shown bellow.

- Visuals:

![sample_1](https://github.com/muslll/neosr/assets/132400428/70f2a38c-7480-43b4-bbb9-84748d2ec260)
![sample_0](https://github.com/muslll/neosr/assets/132400428/6a4f869b-f43a-4832-a88d-5d724ff65984)

- Metrics (Adan-SF - Green | AdamW-SF Blue):
```mermaid
xychart-beta
    title "Adan vs AdamW Schedule-Free"
    x-axis [5k, 10k, 15k, 20k, 25k, 30k, 35k, 40k, 60k, 80k, 100k, 120k, 140k, 160k, 180k]
    y-axis "SSIM (higher is better)"
    line [0.6985391491719667, 0.7079623345237144, 0.709300201535295, 0.7101285333764074, 0.7110304413788211, 0.713379663456067, 0.714167268005285, 0.7161271662440858, 0.7157212519234937, 0.7163480334299989, 0.7175525168240516, 0.718333561897245, 0.7164353289949538, 0.7183044089380414, 0.7166775441925853]
    line [0.7028505604379692, 0.7098506185652865, 0.712317121415303, 0.7137367388139673, 0.7117023167912105, 0.7142965463942005, 0.7138159364238664, 0.7132726319085899, 0.7151120558535241, 0.7167295251201583, 0.7175141784804421, 0.7158511167152422, 0.7171361890123437, 0.7178759658614748, 0.7179985009511379]

```

## license and acknowledgements

Released under [Apache 2.0](https://github.com/muslll/adan_schedule_free/license). Code adapted from official repositories [Adan](https://github.com/sail-sg/Adan) and [Schedule-Free](https://github.com/facebookresearch/schedule_free).

Original research papers:
```bibtex

@article{xie2022adan,
  title={Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models},
  author={Xie, Xingyu and Zhou, Pan and Li, Huan and Lin, Zhouchen and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2208.06677},
  eprint={2208.06677},
  year={2022},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2208.06677}
}

@article{defazio2024road,
  title={The Road Less Scheduled},
  author={Aaron Defazio and Xingyu Yang and Harsh Mehta and Konstantin Mishchenko and Ahmed Khaled and Ashok Cutkosky},
  journal={arXiv preprint arXiv:2405.15682},
  eprint={2405.15682},
  year={2024},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2405.15682}
}
```

## support me

> [!TIP]
> Consider supporting me on [**KoFi**](https://ko-fi.com/muslll) &#9749; or [**Patreon**](https://www.patreon.com/neosr)
