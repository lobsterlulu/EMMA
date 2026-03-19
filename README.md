# EMMA

Evaluation toolkit for measuring concept erasure in text-to-image diffusion models. Provides classifiers for detecting whether target concepts appear in generated images, and prompt sets for systematically probing erasure across multiple domains.

## Components

### [`classifier/`](classifier/)

Inference-only classifiers for evaluating concept erasure across five domains: art style, celebrity, copyright, NSFW, and object. Includes 6 classifiers (ML-Decoder, Giphy Celebrity Detector, UnlearnDiffAtk, NudeNet, Q16, YOLOv11).

See [`classifier/README.md`](classifier/README.md) for setup and usage details.

### [`prompts/`](prompts/)

Domain-specific prompt variations for testing erasure robustness. Covers 5 domains with up to 7 variation types per domain (direct name, prefix, synonym variant, short, long, random, and adversarial).

See [`prompts/README.md`](prompts/README.md) for the full prompt taxonomy.
