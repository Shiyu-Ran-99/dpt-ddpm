from copy import deepcopy
import torch
import os
import numpy as np
import zero
from dpt_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset, update_ema
import lib
import pandas as pd
from opacus import PrivacyEngine

class Trainer:
    def __init__(self, diffusion, train_iter, lr, lr_dp, weight_decay, steps, device=torch.device('cpu')):
        
        # 2024-11-13 Shiyu add
        self._delta = 1e-5
        self.epsilon_target = 70.0
        print(f"epsilon_target is {self.epsilon_target}")
        self.steps = steps
        self.init_lr = lr
        self.lr_dp = lr_dp
        self.diffusion = diffusion
        self.train_iter = train_iter
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=self.init_lr, weight_decay=weight_decay)
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss', 'iloss', 'oloss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)


    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        # loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss_multi, loss_gauss, loss_info, loss_outlier = self.diffusion(x, out_dict)
        loss = loss_multi + loss_gauss + loss_info + loss_outlier #+ wasserstein_distance(x, out_dict)
        # print(f"loss is {loss}")
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss, loss_info, loss_outlier

    def trainer(self, count):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_info = 0.0
        curr_loss_outlier = 0.0

        curr_count = 0
        while step < self.steps:
            for x, out_dict in self.train_iter:

                self._eps = self.privacy_engine.get_epsilon(self._delta)
                # print(f"eps is {self._eps}")
                if self._eps >= self.epsilon_target:
                    print(f"Privacy budget reached in step {step}.")
                    return self

                batch_loss_multi, batch_loss_gauss, batch_loss_info, batch_loss_outlier = self._run_step(x, out_dict)

                self._anneal_lr(step)

                curr_count += len(x)
                curr_loss_multi += batch_loss_multi.item() * len(x)
                curr_loss_gauss += batch_loss_gauss.item() * len(x)
                curr_loss_info += batch_loss_info.item() * len(x)
                curr_loss_outlier += batch_loss_outlier.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                iloss = np.around(curr_loss_info / curr_count, 4)
                oloss = np.around(curr_loss_outlier / curr_count, 4)
                if count > 1:
                    if mloss + gloss + iloss +oloss > self.last_loss_multi + self.last_loss_gauss + self.last_loss_info + self.last_loss_outlier:
                        print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} ILoss: {iloss} OLoss: {oloss} Sum: {mloss + gloss + iloss + oloss}')
                        break
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} ILoss: {iloss} OLoss: {oloss} Sum: {mloss + gloss + iloss + oloss}')
                self.loss_history.loc[len(self.loss_history)] = [step + 1, mloss, gloss, iloss, oloss, mloss + gloss + iloss + oloss]
                self.last_loss_multi, self.last_loss_gauss, self.last_loss_info, self.last_loss_outlier = mloss, gloss, iloss, oloss
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0
                curr_loss_info = 0.0
                curr_loss_outlier = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())
            step += 1

    def run_loop(self):
        count = 0
        # train without DP
        count += 1
        self.last_loss_multi = float('inf')
        self.last_loss_gauss = float('inf')
        self.last_loss_info = float('inf')
        self.last_loss_outlier = float('inf')
        self.trainer(count)

        # # add DP_SGD module
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=self.lr_dp)
        self.diffusion, self.optimizer, self.train_iter = self.privacy_engine.make_private_with_epsilon(
            module=self.diffusion,
            optimizer=self.optimizer,
            data_loader=self.train_iter,
            target_epsilon=self.epsilon_target,
            target_delta=self._delta,
            epochs=self.steps,
            max_grad_norm=1.0,
            poisson_sampling=True,
            # grad_sample_mode="functorch"
        )
        # train with DP
        count += 1
        self.last_loss_multi = float('inf')
        self.last_loss_gauss = float('inf')
        self.last_loss_info = float('inf')
        self.last_loss_outlier = float('inf')
        self.trainer(count)

        print(f"After training, eps is {self._eps}")


def train(
        parent_dir,
        real_data_path='data/higgs-small',
        steps=1000,
        lr=0.002,
        lr_dp=0.001,
        weight_decay=1e-4,
        batch_size=1024,
        model_type='mlp',
        model_params=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        T_dict=None,
        num_numerical_features=0,
        device=torch.device('cuda:1'),
        seed=0,
        change_val=False
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )
    print(f"dataset is {real_data_path.split('/')[1]}")

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(f"categorical encoding's size is {K}")

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    print(f"num_numerical_features is {num_numerical_features}")
    d_in = np.sum(K) + num_numerical_features
    if len(model_params['embedding_type']) != 0:
        d_embedding_in = np.sum(K) + num_numerical_features * model_params['d_embedding']
    else:
        d_embedding_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = int(d_embedding_in)

    print(f"model input dimension is {d_embedding_in}")

    print(model_params)
    print(f"steps is: {steps}, lr is {lr}, lr_dp is {lr_dp}, weight_decay is {weight_decay}, batch_size is {batch_size}, model_type is {model_type}")
    # embedding_type = 'LinearEmbeddings'
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        d_out=d_in,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    train_loader = lib.prepare_torch_dataloader(dataset, split='train', batch_size=batch_size, shuffle=False)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device,
        embedding_type=model_params['embedding_type'],
        d_embedding=model_params['d_embedding']
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_iter=train_loader,
        lr=lr,
        lr_dp=lr_dp,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))