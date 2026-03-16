"""WER evaluation for distilled student models.

Reuses the evaluation recipe from omnilingual-asr with the student checkpoint.

Usage:
    python scripts/evaluate.py <output_dir> --config-file configs/eval.yaml
"""

from fairseq2.recipe.cli import eval_main

from omnilingual_asr.workflows.recipes.wav2vec2.asr.eval.recipe import (
    Wav2Vec2AsrEvalRecipe,
)

recipe = Wav2Vec2AsrEvalRecipe()
eval_main(recipe)
