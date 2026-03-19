# Prompts

Domain-specific prompt sets for evaluating concept erasure in text-to-image models. Each domain directory contains JSON files with prompt variations designed to test whether a model has successfully unlearned a target concept.

## Domains

| Directory | Description |
|-----------|-------------|
| `art/` | Art style prompts targeting specific artists |
| `celebrity/` | Celebrity identity prompts |
| `copyright/` | Brand/logo copyright prompts |
| `nsfw/` | NSFW/inappropriate content prompts |
| `object/` | COCO 79 object prompts (exclude *person*)  |

## Prompt Variation Types

Each domain contains up to 7 JSON files, each representing a different prompting strategy:

| File | Type | Dimension | Description |
|------|------|------|-------------|
| `1_name.json` | Name | Erasing ability | Direct concept name |
| `2_prefix.json` | Prefix | Erasing ability |Adds contextual prefixes to the concept name |
| `3_variant.json` | Variant | Erasing ability |Synonym and alias substitutions for the concept |
| `4_short.json` | Short | Erasing ability |Concise prompts |
| `5_long.json` | Long | Erasing ability |Detailed, descriptive prompts |
| `6_random.json` | Random | Retaining ability | Randomly selected non-target concepts |
| `7_hard.json` | Hard | Retaining ability | Visually similar non-target concepts |

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
