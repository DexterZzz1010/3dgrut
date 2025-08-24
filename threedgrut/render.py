# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import create_summary_writer


class Renderer:
    def __init__(
        self, model, conf, global_step, out_dir, path="", save_gt=True, writer=None, compute_extra_metrics=True, extended_features_metrics=False
    ) -> None:

        if path:  # Replace the path to the test data
            conf.path = path

        self.model = model
        self.out_dir = out_dir
        self.save_gt = save_gt
        self.path = path
        self.conf = conf
        self.global_step = global_step
        self.dataset, self.dataloader = self.create_test_dataloader(conf)
        self.writer = writer
        self.compute_extra_metrics = compute_extra_metrics
        self.extended_features_metrics = extended_features_metrics

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

    def create_test_dataloader(self, conf):
        """Create the test dataloader for the given configuration."""
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        dataset = datasets.make_test(name=conf.dataset.type, config=conf)
        
        # Configure DataLoader arguments for the current platform
        dataloader_kwargs = configure_dataloader_for_platform({
            'num_workers': 8,
            'batch_size': 1,
            'shuffle': False,
            'collate_fn': None,
        })
        
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        return dataset, dataloader

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path, out_dir, path="", save_gt=True, writer=None, model=None, computes_extra_metrics=True
    ):
        """Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the
        """

        checkpoint = torch.load(checkpoint_path)
        global_step = checkpoint["global_step"]

        conf = checkpoint["config"]
        # overrides
        if conf["render"]["method"] == "3dgrt":
            conf["render"]["particle_kernel_density_clamping"] = True
            conf["render"]["min_transmittance"] = 0.03
        conf["render"]["enable_kernel_timings"] = True

        object_name = Path(conf.path).stem
        experiment_name = conf["experiment_name"]
        writer, out_dir, run_name = create_summary_writer(conf, object_name, out_dir, experiment_name, use_wandb=False)

        if model is None:
            # Initialize the model and the optix context
            model = MixtureOfGaussians(conf)
            # Initialize the parameters from checkpoint
            model.init_from_checkpoint(checkpoint)
        model.build_acc()

        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=computes_extra_metrics,
            extended_features_metrics=extended_features_metrics,
        )

    @classmethod
    def from_preloaded_model(
        cls, model, out_dir, path="", save_gt=True, writer=None, global_step=None, compute_extra_metrics=False, extended_features_metrics=False
    ):
        """Loads checkpoint for test path."""

        conf = model.conf
        if global_step is None:
            global_step = ""
        model.build_acc()
        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=compute_extra_metrics,
            extended_features_metrics=extended_features_metrics,
        )

    @torch.no_grad()
    def render_all(self):
        """Render all the images in the test dataset and log the metrics."""

        # Criterions that we log during training
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}

        if self.compute_extra_metrics:
            criterions |= {
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
            }
        
        # Extended features criterions
        if self.extended_features_metrics:
            criterions |= {
                "psnr_ext": PeakSignalNoiseRatio().to("cuda"),
            }

        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
        os.makedirs(output_path_renders, exist_ok=True)

        if self.save_gt:
            output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt")
            os.makedirs(output_path_gt, exist_ok=True)

        # Extended features output directories
        if self.extended_features_metrics:
            output_path_renders_ext = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders_ext")
            os.makedirs(output_path_renders_ext, exist_ok=True)
            
            if self.save_gt:
                output_path_gt_ext = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt_ext")
                os.makedirs(output_path_gt_ext, exist_ok=True)

        psnr = []
        ssim = []
        lpips = []
        inference_time = []
        test_images = []

        # Extended features metrics tracking
        psnr_ext = []
        test_images_ext = []

        best_psnr = -1.0
        worst_psnr = 2**16 * 1.0
        best_psnr_ext = -1.0
        worst_psnr_ext = 2**16 * 1.0

        best_psnr_img = None
        best_psnr_img_gt = None
        best_psnr_img_ext = None
        best_psnr_img_gt_ext = None

        worst_psnr_img = None
        worst_psnr_img_gt = None
        worst_psnr_img_ext = None
        worst_psnr_img_gt_ext = None

        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")

        for iteration, batch in enumerate(self.dataloader):

            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)

            # Compute the outputs of a single batch
            outputs = self.model(gpu_batch)

            pred_rgb_full = outputs["pred_rgb"]
            rgb_gt_full = gpu_batch.rgb_gt

            # The values are already alpha composited with the background
            torchvision.utils.save_image(
                pred_rgb_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_renders, "{0:05d}".format(iteration) + ".png"),
            )
            pred_img_to_write = pred_rgb_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.writer is not None:
                test_images.append(pred_img_to_write)

            if self.save_gt:
                torchvision.utils.save_image(
                    rgb_gt_full.squeeze(0).permute(2, 0, 1),
                    os.path.join(output_path_gt, "{0:05d}".format(iteration) + ".png"),
                )

            # Compute the loss
            psnr_single_img = criterions["psnr"](outputs["pred_rgb"], gpu_batch.rgb_gt).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            logger.info(f"Frame {iteration}, PSNR: {psnr[-1]}")

            if psnr_single_img > best_psnr:
                best_psnr = psnr_single_img
                best_psnr_img = pred_img_to_write
                best_psnr_img_gt = gt_img_to_write

            if psnr_single_img < worst_psnr:
                worst_psnr = psnr_single_img
                worst_psnr_img = pred_img_to_write
                worst_psnr_img_gt = gt_img_to_write

            # Extended features metrics computation
            if self.extended_features_metrics and "pred_extended_features" in outputs and gpu_batch.features_gt is not None:
                pred_extended_features = outputs["pred_extended_features"]
                extended_features_gt = gpu_batch.features_gt
                
                # Determine the number of feature components to use (minimum between pred and gt)
                n_extended_features = min(pred_extended_features.shape[-1], extended_features_gt.shape[-1])
                
                if n_extended_features > 0:
                    # Use all feature components for PSNR computation
                    pred_ext_all = pred_extended_features[..., :n_extended_features]
                    gt_ext_all = extended_features_gt[..., :n_extended_features]
                    
                    # Resize predicted features to match ground truth size if needed (same as trainer.py)
                    if pred_ext_all.shape != gt_ext_all.shape:
                        pred_ext_all = torch.nn.functional.interpolate(
                            pred_ext_all.permute(0, 3, 1, 2),  # [B, C, H, W]
                            size=gt_ext_all.shape[1:3],
                            mode='area'
                        ).permute(0, 2, 3, 1)  # Back to [B, H, W, C]
                    
                    # Compute PSNR using all feature components
                    psnr_single_img_ext = criterions["psnr_ext"](pred_ext_all, gt_ext_all).item()
                    psnr_ext.append(psnr_single_img_ext)
                    
                    # For visualization, use only first 3 components if available
                    if n_extended_features >= 3:
                        gt_ext_rgb = gt_ext_all[..., :3]
                        # min-max normalize
                        gt_ext_rgb_min, gt_ext_rgb_max = gt_ext_rgb.aminmax()
                        gt_ext_rgb = (gt_ext_rgb - gt_ext_rgb_min) / (gt_ext_rgb_max - gt_ext_rgb_min)
                        
                        pred_ext_rgb = pred_ext_all[..., :3]
                        # min-max normalize
                        pred_ext_rgb_min, pred_ext_rgb_max = pred_ext_rgb.aminmax()
                        pred_ext_rgb = (pred_ext_rgb - pred_ext_rgb_min) / (pred_ext_rgb_max - pred_ext_rgb_min)
                        
                        # The values are already alpha composited with the background
                        torchvision.utils.save_image(
                            pred_ext_rgb.squeeze(0).permute(2, 0, 1),
                            os.path.join(output_path_renders_ext, "{0:05d}".format(iteration) + ".png"),
                        )
                        pred_img_ext_to_write = pred_ext_rgb[-1]
                        gt_img_ext_to_write = gt_ext_rgb[-1]

                        if self.writer is not None:
                            test_images_ext.append(pred_img_ext_to_write)

                        if self.save_gt:
                            torchvision.utils.save_image(
                                gt_ext_rgb.squeeze(0).permute(2, 0, 1),
                                os.path.join(output_path_gt_ext, "{0:05d}".format(iteration) + ".png"),
                            )
                        
                        if psnr_single_img_ext > best_psnr_ext:
                            best_psnr_ext = psnr_single_img_ext
                            best_psnr_img_ext = pred_img_ext_to_write
                            best_psnr_img_gt_ext = gt_img_ext_to_write

                        if psnr_single_img_ext < worst_psnr_ext:
                            worst_psnr_ext = psnr_single_img_ext
                            worst_psnr_img_ext = pred_img_ext_to_write
                            worst_psnr_img_gt_ext = gt_img_ext_to_write

            # evaluate on full image
            ssim.append(
                criterions["ssim"](
                    pred_rgb_full.permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )
            lpips.append(
                criterions["lpips"](
                    pred_rgb_full.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )

            # Record the time
            inference_time.append(outputs["frame_time_ms"])

            logger.log_progress(task_name="Rendering", advance=1, iteration=f"{str(iteration)}", psnr=psnr[-1])

        logger.end_progress(task_name="Rendering")

        mean_psnr = np.mean(psnr)
        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips)
        std_psnr = np.std(psnr)
        mean_inference_time = np.mean(inference_time)

        table = dict(
            mean_psnr=mean_psnr,
            mean_ssim=mean_ssim,
            mean_lpips=mean_lpips,
            std_psnr=std_psnr,
        )

        # Extended features metrics
        if self.extended_features_metrics and len(psnr_ext) > 0:
            mean_psnr_ext = np.mean(psnr_ext)
            std_psnr_ext = np.std(psnr_ext)
            table["mean_psnr_ext"] = mean_psnr_ext
            table["std_psnr_ext"] = std_psnr_ext

        if self.conf.render.enable_kernel_timings:
            table["mean_inference_time"] = f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"

        logger.log_table(f"⭐ Test Metrics - Step {self.global_step}", record=table)

        if self.writer is not None:
            self.writer.add_scalar("psnr/test", mean_psnr, self.global_step)
            self.writer.add_scalar("ssim/test", mean_ssim, self.global_step)
            self.writer.add_scalar("lpips/test", mean_lpips, self.global_step)
            self.writer.add_scalar("time/inference/test", mean_inference_time, self.global_step)
            
            # Extended features metrics logging
            if self.extended_features_metrics and len(psnr_ext) > 0:
                self.writer.add_scalar("psnr_ext/test", mean_psnr_ext, self.global_step)

            if len(test_images) > 0:
                self.writer.add_images(
                    "image/pred/test",
                    torch.stack(test_images),
                    self.global_step,
                    dataformats="NHWC",
                )

            # Extended features image logging
            if self.extended_features_metrics and len(test_images_ext) > 0:
                self.writer.add_images(
                    "image/pred_ext/test",
                    torch.stack(test_images_ext),
                    self.global_step,
                    dataformats="NHWC",
                )

            if best_psnr_img is not None:
                self.writer.add_images(
                    "image/best_psnr/test",
                    torch.stack([best_psnr_img, best_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if worst_psnr_img is not None:
                self.writer.add_images(
                    "image/worst_psnr/test",
                    torch.stack([worst_psnr_img, worst_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            # Extended features best/worst image logging
            if self.extended_features_metrics and best_psnr_img_ext is not None:
                self.writer.add_images(
                    "image/best_psnr_ext/test",
                    torch.stack([best_psnr_img_ext, best_psnr_img_gt_ext]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if self.extended_features_metrics and worst_psnr_img_ext is not None:
                self.writer.add_images(
                    "image/worst_psnr_ext/test",
                    torch.stack([worst_psnr_img_ext, worst_psnr_img_gt_ext]),
                    self.global_step,
                    dataformats="NHWC",
                )

        return mean_psnr, std_psnr, mean_inference_time
