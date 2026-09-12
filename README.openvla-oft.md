# AURA-VLA: Adaptive-Uncertainty and Region-Alignment Optimized Fine-Tuning for Vision-Language-Action Models

AURA-VLA is a fine-tuning framework for Vision-Language-Action (VLA) models. It extends the OpenVLA/OFT training stack with two auxiliary objectives:

- **Adaptive uncertainty:** predicts action risk from action-token representations and action statistics.
- **Region alignment:** aligns language-conditioned visual patch features with target regions, optionally using negative-region contrast and counterfactual action supervision.

The implementation retains the OpenVLA-compatible continuous action heads (L1 regression, diffusion, and flow matching), LoRA fine-tuning, and LIBERO/ALOHA workflows. The AURA-VLA additions are opt-in, so ordinary OpenVLA fine-tuning remains available.

## Repository layout

- `vla-scripts/finetune.py` — main fine-tuning entry point and AURA-VLA configuration.
- `prismatic/models/aura_vla.py` — adaptive-uncertainty and region-alignment module.
- `experiments/robot/libero/run_libero_eval.py` — LIBERO evaluation entry point.
- `experiments/robot/openvla_utils.py` and `experiments/robot/robot_utils.py` — model loading and action-generation utilities.
- `LIBERO.md` and `ALOHA.md` — benchmark-specific setup and commands.

## Installation

Create a Python 3.10 environment, install PyTorch for your CUDA setup, then install this repository and Flash Attention 2:

```bash
conda create -n aura-vla python=3.10 -y
conda activate aura-vla
pip install torch torchvision torchaudio
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

See [SETUP.md](SETUP.md) for the original environment notes. For LIBERO, also install the benchmark and its extra requirements:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

## Fine-tuning

The following is a standard L1-regression fine-tuning command. Replace paths, dataset name, GPU count, and Weights & Biases values for your environment.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /PATH/TO/RUNS \
  --use_l1_regression True \
  --use_diffusion False \
  --use_flow_matching False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --max_steps 150005 \
  --save_freq 10000 \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity YOUR_WANDB_ENTITY \
  --wandb_project aura-vla
```

### Enable AURA-VLA

Add `--use_aura_vla True` to enable the auxiliary module. These arguments tune its losses:

| Argument | Default | Purpose |
| --- | ---: | --- |
| `--aura_hidden_dim` | model hidden size | Hidden width of the AURA-VLA heads. |
| `--aura_uncertainty_loss_weight` | `1.0` | Weight of uncertainty calibration. |
| `--aura_uncertainty_error_scale` | `1.0` | L1-error scale used to form risk targets. |
| `--region_alignment_loss_weight` | `1.0` | Weight of BCE + Dice region alignment. |
| `--region_alignment_dice_weight` | `1.0` | Dice-loss multiplier within region alignment. |
| `--region_contrastive_loss_weight` | `1.0` | Weight of negative-region contrast. |
| `--region_equivariance_loss_weight` | `1.0` | Weight of counterfactual action consistency. |
| `--region_contrastive_temperature` | `0.07` | Contrastive-softmax temperature. |

For region-supervised training, batches may provide `target_masks` and optional `negative_target_masks`. Counterfactual supervision is optional and uses the existing `counterfactual_input_ids`, `counterfactual_labels`, and `counterfactual_target_actions` (or `counterfactual_actions`) batch fields. When those fields are absent, their corresponding auxiliary terms are skipped; uncertainty calibration remains active when predicted and ground-truth actions are available.

Each AURA-VLA checkpoint includes a separate `aura_vla_module--<step>_checkpoint.pt` file alongside the action head, projector, and model checkpoints. Keep these files together when resuming or using AURA-VLA inference helpers.

## Evaluation

After installing LIBERO, run a task suite with a compatible checkpoint:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /PATH/TO/CHECKPOINT \
  --task_suite_name libero_spatial \
  --center_crop True
```

Use the same action-head mode, number of input images, proprioception setting, and LoRA rank that were used during training. See [LIBERO.md](LIBERO.md) for dataset acquisition, suite-specific commands, and evaluation details; see [ALOHA.md](ALOHA.md) for the real-robot workflow.

## Programmatic loading

`get_aura_vla_module` in `experiments/robot/openvla_utils.py` loads the separate AURA-VLA checkpoint. Pass the resulting module to the action-generation utilities as `aura_vla_module` when using adaptive flow-matching inference. The helper looks for the `aura_vla_module` checkpoint name documented above.

## Citation

If you use this code, please cite the accompanying AURA-VLA paper:

```bibtex
@misc{aura_vla,
  title={AURA-VLA: Adaptive-Uncertainty and Region-Alignment Optimized Fine-Tuning for Vision-Language-Action Models},
  note={Project paper}
}
```
