"""Entry point for Stage 1 distillation (size reduction)."""

from fairseq2.recipe.cli import train_main

from omni_asr_distil.distill_recipe import DistillRecipe

recipe = DistillRecipe()
train_main(recipe)
