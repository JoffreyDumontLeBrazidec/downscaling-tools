import torch


class SigmaEvaluator:
    def __init__(self, downscaler, datamodule, N_samples):
        self.downscaler = downscaler
        self.datamodule = datamodule
        self.N_samples = N_samples

    def evaluate_sigma(self, sigma, prediction_on_pure_noise):
        self.downscaler.eval()
        total_loss = 0.0
        total_metrics = {}
        n_batches = 0

        dataloader = self.datamodule.val_dataloader()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.N_samples:
                break
            loss, metrics = self.evaluate_batch_with_sigma(
                sigma, batch, prediction_on_pure_noise
            )
            total_loss += loss.item()

            for key, value in metrics.items():
                if torch.is_tensor(value):
                    value = value.detach().float().cpu().item()
                total_metrics[key] = total_metrics.get(key, 0.0) + float(value)

            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_metrics = {key: value / n_batches for key, value in total_metrics.items()}

        return avg_loss, avg_metrics

    def evaluate_batch_with_sigma(self, sigma, batch, prediction_on_pure_noise):
        with torch.inference_mode():
            sigma = torch.Tensor([sigma]).to(self.downscaler.device)
            sigma_view = sigma.view(-1, 1, 1, 1)
            batch = [x.to(self.downscaler.device) for x in batch]
            x_in, x_in_hres, y = batch

            x_in_interp_to_hres = (
                self.downscaler.model.model.apply_interpolate_to_high_res(
                    x_in[:, 0, ...],
                    grid_shard_shapes=self.downscaler.lres_grid_shard_shapes,
                    model_comm_group=self.downscaler.model_comm_group,
                )[:, None, ...]
            )

            self.downscaler.x_in_matching_channel_indices = (
                self.downscaler.x_in_matching_channel_indices.to(
                    x_in_interp_to_hres.device
                )
            )
            residuals_target = self.downscaler.model.model.compute_residuals(
                y,
                x_in_interp_to_hres[..., self.downscaler.x_in_matching_channel_indices],
            )

            x_in_interp_to_hres = self.downscaler.model.pre_processors(
                x_in_interp_to_hres, dataset="input_lres"
            )
            x_in_hres = self.downscaler.model.pre_processors(
                x_in_hres, dataset="input_hres"
            )
            residuals_target = self.downscaler.model.pre_processors(
                residuals_target, dataset="output"
            )

            # get noise level and associated loss weights
            sigma_data = 1
            noise_weights = (sigma_view**2 + sigma_data**2) / (
                sigma_view * sigma_data
            ) ** 2
            if prediction_on_pure_noise:
                residuals_target_noised = (
                    torch.randn(
                        residuals_target.shape,
                        device=residuals_target.device,
                    )
                    * sigma
                )
            elif not prediction_on_pure_noise:
                residuals_target_noised = self.downscaler._noise_target(
                    residuals_target, sigma
                )

            y_pred = self.downscaler(
                x_in_interp_to_hres,
                x_in_hres,
                residuals_target_noised,
                sigma_view,
            )
            loss, metrics_next = self.downscaler.compute_loss_metrics(
                y_pred=y_pred[:, 0, ...],
                y=residuals_target[:, 0, ...],
                rollout_step=0,
                training_mode=True,
                validation_mode=True,
                weights=noise_weights,
                use_reentrant=False,
            )
            denorm_pred_residuals = self.downscaler.model.post_processors(
                y_pred, dataset="output", in_place=False
            )
            denorm_truth_residuals = self.downscaler.model.post_processors(
                residuals_target[:, 0, ...], dataset="output", in_place=False
            )

            diff = denorm_pred_residuals - denorm_truth_residuals

            metrics_next["diff_all_var_non_weighted"] = torch.sqrt(torch.mean(diff**2))
            FIELD_IDX = {
                "10u": 0,
                "10v": 1,
                "2d": 2,
                "2t": 3,
                "msl": 4,
                "skt": 15,
                "sp": 16,
                "tcw": 27,
                "z_500": 64,
                "u_850": 36,
                "v_850": 46,
            }

            for name, idx in FIELD_IDX.items():
                metrics_next[f"mse_{name}_non_weighted"] = torch.mean(
                    diff[..., idx] ** 2
                )

            del y_pred, residuals_target_noised, x_in, x_in_hres, residuals_target
        return loss, metrics_next
