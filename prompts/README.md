# Prompts

Domain-specific prompt sets for evaluating concept erasure in text-to-image models. Each domain directory contains JSON files with prompt variations designed to test whether a model has successfully unlearned a target concept.

## Domains

| Directory | Description |
|-----------|-------------|
| `art/` | Art style prompts targeting specific artists (e.g., Salvador Dali, Andy Warhol) |
| `celebrity/` | Celebrity identity prompts |
| `copyright/` | Brand/logo copyright prompts |
| `nsfw/` | NSFW/inappropriate content prompts |
| `object/` | COCO object class prompts |

## Prompt Variation Types

Each domain contains up to 7 JSON files, each representing a different prompting strategy:

| File | Type | Description |
|------|------|-------------|
| `1_name.json` | Name | Direct concept name (baseline) |
| `2_prefix.json` | Prefix | Adds contextual prefixes to the concept name |
| `3_variant.json` | Variant | Synonym and alias substitutions for the concept |
| `4_short.json` | Short | Concise, minimal prompts |
| `5_long.json` | Long | Detailed, descriptive prompts |
| `6_random.json` | Random | Randomly constructed prompt variations |
| `7_hard.json` | Hard | Adversarial prompts designed to bypass erasure |

> **Note:** `art/` and `celebrity/` do not include `3_variant.json`.

## File Format

Each JSON file is an array of objects. Each object maps a concept key to a list of prompt strings:

```json
[
  {
    "concept-key": [
      "prompt string 1",
      "prompt string 2"
    ]
  }
]
```
