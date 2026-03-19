# EMMA

[![arXiv](https://img.shields.io/badge/arXiv-2512.17320-b31b1b.svg)](https://arxiv.org/abs/2512.17320)

Bechmark for measuring concept erasure in text-to-image diffusion models. Provides classifiers for detecting whether target/non-target concepts appear in generated images, and prompt sets for systematically probing erasure across multiple domains.

## Components

### [`classifier/`](classifier/)

Classifiers for evaluating concept erasure across five domains: art style, celebrity, copyright, NSFW, and object. Includes 6 classifiers (ML-Decoder, GCD, UnlearnDiffAtk, NudeNet, Q16, YOLOv11).

See [`classifier/README.md`](classifier/README.md) for setup and usage details.

### [`prompts/`](prompts/)

Prompt variations for testing erasing and retaining ability. Covers 5 domains with up to 7 variation types per domain (direct name, prefix, variant, short, long, random, and hard).

See [`prompts/README.md`](prompts/README.md) for the full prompt taxonomy.
