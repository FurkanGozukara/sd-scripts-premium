import argparse
from typing import List, Optional, Union

import torch
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import sdxl_model_util, sdxl_train_util, strategy_base, strategy_sd, strategy_sdxl, train_util, compile_utils
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class SdxlNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True
        self.is_swapping_blocks = False

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        # Verify blocks_to_swap compatibility
        if hasattr(args, 'blocks_to_swap') and args.blocks_to_swap and args.blocks_to_swap > 0:
            if not (args.fused_backward_pass or (hasattr(args, 'blockwise_fused_optimizers') and args.blockwise_fused_optimizers)):
                logger.warning(
                    "blocks_to_swap is set but neither fused_backward_pass nor blockwise_fused_optimizers is enabled. "
                    "Block swap works best with fused_backward_pass. Proceeding anyway, but you may want to enable it."
                )

        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def get_tokenize_strategy(self, args):
        return strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy):
        return [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sdxl.SdxlTextEncodingStrategy()

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return text_encoders + [accelerator.unwrap_model(text_encoders[-1])]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, args.skip_cache_check, is_weighted=args.weighted_captions
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders + [accelerator.unwrap_model(text_encoders[-1])], accelerator)
            accelerator.wait_for_everyone()

            text_encoders[0].to("cpu", dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            text_encoders[1].to("cpu", dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            with torch.enable_grad():
                # Get the text embedding for conditioning
                # TODO support weighted captions
                # if args.weighted_captions:
                #     encoder_hidden_states = get_weighted_text_embeddings(
                #         tokenizer,
                #         text_encoder,
                #         batch["captions"],
                #         accelerator.device,
                #         args.max_token_length // 75 if args.max_token_length else 1,
                #         clip_skip=args.clip_skip,
                #     )
                # else:
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                    args.max_token_length,
                    input_ids1,
                    input_ids2,
                    tokenizers[0],
                    tokenizers[1],
                    text_encoders[0],
                    text_encoders[1],
                    None if not args.full_fp16 else weight_dtype,
                    accelerator=accelerator,
                )
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

            # # verify that the text encoder outputs are correct
            # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
            #     args.max_token_length,
            #     batch["input_ids"].to(text_encoders[0].device),
            #     batch["input_ids2"].to(text_encoders[0].device),
            #     tokenizers[0],
            #     tokenizers[1],
            #     text_encoders[0],
            #     text_encoders[1],
            #     None if not args.full_fp16 else weight_dtype,
            # )
            # b_size = encoder_hidden_states1.shape[0]
            # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # logger.info("text encoder outputs verified")

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        indices: Optional[List[int]] = None,
    ):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        if indices is not None and len(indices) > 0:
            noisy_latents = noisy_latents[indices]
            timesteps = timesteps[indices]
            text_embedding = text_embedding[indices]
            vector_embedding = vector_embedding[indices]

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)
    
    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        """
        Called after validation step. Used to prepare block swap for next forward pass.
        """
        if self.is_swapping_blocks:
            # Prepare for next forward: because backward pass is not called during validation,
            # we need to prepare block swap here
            unwrapped = accelerator.unwrap_model(unet)
            if hasattr(unwrapped, 'block_swap_manager'):
                unwrapped.block_swap_manager.prepare_block_swap_before_forward()

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Prepare UNet with accelerator, optionally with block swap and/or compile.
        """
        # Check if block swapping is enabled
        blocks_to_swap_count = args.blocks_to_swap if hasattr(args, 'blocks_to_swap') and args.blocks_to_swap else 0
        self.is_swapping_blocks = blocks_to_swap_count > 0
        
        if not self.is_swapping_blocks:
            # Standard path without block swapping
            unet = super().prepare_unet_with_accelerator(args, accelerator, unet)
            
            # Then compile if requested
            if hasattr(args, 'compile') and args.compile:
                logger.info("Compiling SDXL UNet blocks with torch.compile")
                unwrapped_unet = accelerator.unwrap_model(unet)
                # Get all block lists for SDXL UNet
                target_blocks = [
                    unwrapped_unet.input_blocks,
                    unwrapped_unet.output_blocks,
                    [unwrapped_unet.middle_block]  # middle_block is a list of modules, but treat as single block
                ]
                unet = compile_utils.compile_model(
                    args,
                    unet,
                    target_blocks,
                    disable_linear=False,
                    log_prefix="SDXL UNet (LoRA)"
                )
                # Add _orig_mod reference for accelerator compatibility
                unet.__dict__["_orig_mod"] = unet
            
            return unet
        
        # Block swapping path using OneTrainer-style approach
        logger.info(f"Setting up SDXL UNet with OneTrainer-style block swap: {blocks_to_swap_count} blocks")
        
        # Block swapping works best with gradient checkpointing
        if args.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing with block swapping.")
        else:
            logger.warning("Gradient checkpointing is disabled properly. VRAM usage optimization by block move will be limited.")

        # Import our SDXL offloading utilities
        import library.sdxl_offloading_utils as sdxl_offloading_utils
        
        # Get UNet structure info
        logger.info(f"SDXL UNet structure: {len(unet.input_blocks)} input blocks, "
                   f"1 middle block, {len(unet.output_blocks)} output blocks")
        
        # Collect transformer blocks for OneTrainer-style swapping
        transformer_blocks = sdxl_offloading_utils.collect_transformer_blocks_from_unet(unet)
        logger.info(f"Found {len(transformer_blocks)} transformer blocks to manage")
        
        # Validate blocks_to_swap
        max_swappable = max(0, len(transformer_blocks) - 2)
        if blocks_to_swap_count > max_swappable:
            logger.warning(f"Requested {blocks_to_swap_count} blocks to swap, but only "
                          f"{max_swappable} can be safely swapped.")
            blocks_to_swap_count = min(blocks_to_swap_count, max_swappable)
        
        logger.info(f"Will swap {blocks_to_swap_count} transformer blocks using backward hooks")
        
        # Enable OneTrainer-style block swap
        swap_manager = sdxl_offloading_utils.enable_block_swap_for_sdxl_unet(
            unet,
            blocks_to_swap_count,
            accelerator.device,
            debug=getattr(args, 'debug', False)
        )
        
        # Prepare with accelerator - let accelerator handle device placement
        # For block swap, we MUST disable automatic device placement to avoid loading the whole model to GPU
        unet = accelerator.prepare(unet, device_placement=False)
        
        # Manually place the model parts:
        # 1. Unwrapped UNet (inner model)
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        # 2. Use our utility to place everything correctly (kept blocks -> GPU, swapped blocks -> CPU, others -> GPU)
        sdxl_offloading_utils.place_unet_for_block_swap(
            unwrapped_unet, 
            transformer_blocks, 
            blocks_to_swap_count, 
            accelerator.device
        )
        
        # Prepare block devices for initial forward pass (this might be redundant if place_unet_for_block_swap did it, but safe)
        if hasattr(unwrapped_unet, 'block_swap_manager'):
            unwrapped_unet.block_swap_manager.prepare_block_swap_before_forward()
            logger.info("OneTrainer-style block swap initialized successfully")
        
        # Note: Compilation is not recommended with block swapping
        if hasattr(args, 'compile') and args.compile:
            logger.warning(
                "torch.compile with block swapping is experimental and may cause issues. "
                "Skipping compilation for block-swapped UNet."
            )
        
        return unet


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    compile_utils.add_compile_arguments(parser)
    
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="Use scaled fp8 for SDXL UNet (recommended for VRAM reduction)"
        " / SDXL UNetにスケーリングされたfp8を使う（VRAM削減に推奨）",
    )
    parser.add_argument(
        "--fp8_fast",
        action="store_true",
        help="Enable fast FP8 arithmetic (requires SM 8.9+, RTX 4XXX+), only effective with fp8_scaled"
        " / 高速FP8演算を有効化（SM 8.9+が必要、RTX 4XXX+）、fp8_scaledと併用時のみ有効",
    )
    
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = SdxlNetworkTrainer()
    trainer.train(args)
