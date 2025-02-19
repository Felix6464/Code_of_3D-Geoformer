from Geoformer import Geoformer
from myconfig import mypara
import torch
from torch.utils.data import DataLoader
import numpy as np
import math
from LoadData import make_dataset2, make_testdataset
import os
import swinlstm
import einops
from torch import nn

def get_statistics(prediction: torch.Tensor, dim: int = 1, mode: str = 'ensemble', epsilon: float = 1e-9):
    '''Compute the mean and standard deviation of the predictive distribution

    Author: @jannikthuemmel
    Args:
         prediction: (batch, ensemble, *) tensor of ensemble predictions
         dim: the dimension of the ensemble
         mode: the type of predictive distribution, can be 'ensemble', 'parametric' or 'sample'
         epsilon: a small number to add to the standard deviation to avoid numerical instability
    Returns:
        mu, sigma     (batch, *) tensors of mean and standard deviation
    '''
    #print("prediction: ", prediction.shape)
    if mode == 'ensemble':
        mu, sigma = prediction.mean(dim = dim), prediction.std(dim = dim) #mean and standard deviation of the ensemble
    elif mode == 'parametric':
        mu, sigma = prediction.split(1, dim = dim)
        mu, sigma = mu.squeeze(dim=dim), sigma.squeeze(dim=dim)
    elif mode == 'sample':
        mu, sigma = prediction, torch.ones_like(prediction)
    else:
        raise NotImplementedError(f'Mode {mode} not implemented')

    return mu, sigma + epsilon

class NormalCRPS(nn.Module):
    '''Continuous Ranked Probability Score (CRPS) loss for a normal distribution
    as described in the paper "Probabilistic Forecasting with Gated Neural Networks".
    
    Implementation by @jannikthuemmel
    '''
    def __init__(self, reduction: str = 'mean', dim: int = 1,  
                 mode: str = 'ensemble'):
        '''
        reduction: the reduction method to use, can be 'mean', 'sum' or 'none'
        sigma_transform: the transform to apply to the std estimate, can be 'softplus', 'exp' or 'none'
        '''
        super().__init__()
        self.dim, self.mode = dim, mode

        self.sqrtPi = torch.as_tensor(np.pi).sqrt()
        self.sqrtTwo = torch.as_tensor(2.).sqrt()

        if reduction == 'mean':
            self.reduce = lambda x: x.mean()
        elif reduction == 'sum':
            self.reduce = lambda x: x.sum()
        elif reduction == 'none':
            self.reduce = lambda x: x
        else:
            raise NotImplementedError(f'Reduction {reduction} not implemented')

    def forward(self, observation: torch.Tensor, prediction: torch.Tensor):
        '''
        Compute the CRPS for a normal distribution
            :param observation: (batch, *) tensor of observations
            :param mu: (batch, *) tensor of means
            :param log_sigma: (batch, *) tensor of log standard deviations
            :return: CRPS score     
            '''
        mu, sigma = get_statistics(prediction, mode=self.mode, dim=self.dim)
        #print("mu: ", mu.shape)
        #print("sigma: ", sigma.shape)
        #print("observation: ", observation.shape)

        z = (observation - mu) / sigma #z transform
        phi = torch.exp(-z ** 2 / 2).div(self.sqrtTwo * self.sqrtPi) #standard normal pdf
        score = sigma * (z * torch.erf(z / self.sqrtTwo) + 2 * phi - 1 / self.sqrtPi) #crps as per Gneiting et al 2005
        reduced_score = self.reduce(score)
        return reduced_score

class lrwarm:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
    

class modelTrainer:
    def __init__(self, mypara):
        assert mypara.input_channal == mypara.output_channal
        self.mypara = mypara
        self.device = mypara.device
        self.loss_fn = NormalCRPS(reduction='none', mode=self.mypara.loss_mode, dim=1)
        self.decay = [self.mypara.decay_rate**i for i in range(self.mypara.sequence_length_train)] if self.mypara.decay_rate else [1 for _ in range(self.myparam.sequence_length_train)]
        self.loss_decay = torch.as_tensor(self.decay, device = self.device, dtype = torch.float)

        self.mymodel = model = swinlstm.SwinLSTMNet(
            input_dim = self.mypara.input_dim, 
            output_dim= self.mypara.output_dim,
            num_channels=self.mypara.num_channels,
            num_layers= self.mypara.num_layers,
            patch_size=self.mypara.patch_size_swin,
            num_tails= self.mypara.num_tails,
            k_conv=self.mypara.k_conv,
            num_conditions= self.mypara.num_conditions,
            cutout=self.mypara.cutout,
            step_strided_conv=self.mypara.step_strided_conv,
        ).to(self.device)

        self.optim = torch.optim.AdamW(self.mymodel.parameters(), lr=self.mypara.base_lr,
                            weight_decay=self.mypara.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer=self.optim,
                        max_lr=self.mypara.base_lr,
                        epochs=self.mypara.num_epochs,
                        steps_per_epoch=mypara.iter_epoch,
                        pct_start=self.mypara.pct_start,
                        anneal_strategy='cos',
                        div_factor=self.mypara.div_factor,
                        final_div_factor=self.mypara.final_div_factor)

        # self.adam = torch.optim.Adam(self.mymodel.parameters(), lr=5e-5)
        adam = torch.optim.Adam(self.mymodel.parameters(), lr=0)
        factor = math.sqrt(mypara.d_size * mypara.warmup) * 0.0015
        self.opt = lrwarm(mypara.d_size, factor, mypara.warmup, optimizer=adam)
        self.sstlevel = 0
        if self.mypara.needtauxy:
            self.sstlevel = 2
        ninoweight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
            * np.log(np.arange(24) + 1)
        ).to(mypara.device)
        self.ninoweight = ninoweight[: self.mypara.output_length]

    def calscore(self, y_pred, y_true):
        # compute Nino score
        with torch.no_grad():
            pred = y_pred - y_pred.mean(dim=0, keepdim=True)
            true = y_true - y_true.mean(dim=0, keepdim=True)
            cor = (pred * true).sum(dim=0) / (
                torch.sqrt(torch.sum(pred ** 2, dim=0) * torch.sum(true ** 2, dim=0))
                + 1e-6
            )
            acc = (self.ninoweight * cor).sum()
            rmse = torch.mean((y_pred - y_true) ** 2, dim=0).sqrt().sum()
            sc = 2 / 3.0 * acc - rmse
        return sc.item()

    def loss_var(self, y_pred, y_true):
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])
        rmse = rmse.sqrt().mean(dim=0)
        rmse = torch.sum(rmse, dim=[0, 1])
        return rmse

    def loss_nino(self, y_pred, y_true):
        # with torch.no_grad():
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        return rmse.sum()

    def combien_loss(self, loss1, loss2):
        combine_loss = loss1 + loss2
        return combine_loss

    def model_pred(self, dataloader):
        self.mymodel.eval()
        nino_pred = []
        var_pred = []
        nino_true = []
        var_true = []
        with torch.no_grad():
            for input_var, var_true1 in dataloader:
                input_var = input_var[:, :, :, :48]
                var_true1 = var_true1[:, :, :, :48]

                SST = var_true1[:, :, self.sstlevel]
                nino_true1 = SST[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                #print("Input_var shape: ", input_var.shape)
                input_var = einops.rearrange(input_var, 'b c t h w -> b t c h w')
                out_var = self.mymodel(
                    input_var.float().to(self.device),
                    self.mypara.output_length,
                )
                #print("Output_var shape: ", out_var.shape)
                out_var = einops.rearrange(out_var, 'b x t c h w -> b x c t h w')
                #print("Output_var shape: ", out_var.shape)
                SST_out = out_var[:, :, :, self.sstlevel]
                out_nino = SST_out[
                    :,
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[3, 4])
                var_true.append(var_true1)
                nino_true.append(nino_true1)
                var_pred.append(out_var)
                nino_pred.append(out_nino)
            var_pred = torch.cat(var_pred, dim=0)
            nino_pred = torch.cat(nino_pred, dim=0)
            nino_true = torch.cat(nino_true, dim=0)
            var_true = torch.cat(var_true, dim=0)
            # --------------------
            ninosc = self.calscore(nino_pred[:, 0], nino_true.float().to(self.device))
            loss_var = self.loss_var(var_pred[:, 0], var_true.float().to(self.device)).item()
            loss_nino = self.loss_nino(
                nino_pred[:, 0], nino_true.float().to(self.device)
            ).item()
            #combine_loss = self.combien_loss(loss_var, loss_nino)
            combine_loss = self.loss_fn(var_true.float().to(self.device), var_pred)
            combine_loss = combine_loss.mean().item()

        return (
            var_pred,
            nino_pred,
            loss_var,
            loss_nino,
            combine_loss,
            ninosc,
        )

    def train_model(self, dataset_train, dataset_eval):
        chk_path = self.mypara.model_savepath_swin + f"SwinLSTM_s{self.mypara.seeds}.pkl"
        torch.manual_seed(self.mypara.seeds)
        dataloader_train = DataLoader(
            dataset_train, batch_size=self.mypara.batch_size_train, shuffle=False
        )
        dataloader_eval = DataLoader(
            dataset_eval, batch_size=self.mypara.batch_size_eval, shuffle=False
        )
        count = 0
        best = -math.inf
        sv_ratio = 1
        for i_epoch in range(self.mypara.num_epochs):
            print("==========" * 8)
            print("\n-->epoch: {0}".format(i_epoch))
            # ---------train
            self.mymodel.train()
            for j, (input_var, var_true) in enumerate(dataloader_train):
                #print("j: ", j)
                self.optim.zero_grad()
                input_var = input_var[:, :, :, :48]
                var_true = var_true[:, :, :, :48]
                SST = var_true[:, :, self.sstlevel]
                nino_true = SST[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                if sv_ratio > 0:
                    sv_ratio = max(sv_ratio - 2.5e-4, 0)
                # -------training for one batch
                #print("Input_var shape: ", input_var.shape)
                input_var = einops.rearrange(input_var, 'b c t h w -> b t c h w')
                #print("Input_var shape: ", input_var.shape)
                var_pred = self.mymodel(
                    input_var.float().to(self.device),
                    self.mypara.output_length,
                )
                #print("Output_var shape: ", var_pred.shape)
                #var_pred = einops.rearrange(var_pred[:, 0], 'b c t h w -> b t c h w')
                var_pred = einops.rearrange(var_pred, 'b x t c h w -> b x c t h w')
                #print("Output_var shape: ", var_pred.shape)
                SST_pred = var_pred[:, :, :, self.sstlevel]
                nino_pred = SST_pred[
                    :,
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[3, 4])
                self.opt.optimizer.zero_grad()
                # self.adam.zero_grad()
                loss_var = self.loss_var(var_pred[:, 0], var_true.float().to(self.device))
                loss_nino = self.loss_nino(nino_pred[:, 0], nino_true.float().to(self.device))
                score = self.calscore(nino_pred[:, 0], nino_true.float().to(self.device))
                # loss_var.backward()
                #combine_loss = self.combien_loss(loss_var, loss_nino)
                #Var_pred shape:  torch.Size([8, 2, 20, 9, 48, 120])
                #Var_true shape:  torch.Size([8, 20, 9, 48, 120])
                combine_loss = self.loss_fn(var_true.float().to(self.device), var_pred)
                #Combine_loss:  torch.Size([8, 20, 9, 48, 120])

                if self.mypara.decay == True:
                    combine_loss = combine_loss.sum(dim = (0, 2, 3, 4)) * self.loss_decay[:self.mypara.output_length]

                combine_loss = combine_loss.mean()
                combine_loss.backward()
                #self.opt.step()
                self.optim.step()
                self.scheduler.step()

                # self.adam.step()
                if j % 100 == 0:
                    print(
                        "\n-->batch:{} loss_var:{:.2f}, loss_nino:{:.2f}, score:{:.3f}".format(
                            j, loss_var, loss_nino, score
                        )
                    )

                # ---------Intensive verification
                if (i_epoch + 1 >= 4) and (j + 1) % 200 == 0:
                    (
                        _,
                        _,
                        lossvar_eval,
                        lossnino_eval,
                        comloss_eval,
                        sceval,
                    ) = self.model_pred(dataloader=dataloader_eval)
                    print(
                        "-->Evaluation... \nloss_var:{:.3f} \nloss_nino:{:.3f} \nloss_com:{:.3f} \nscore:{:.3f}".format(
                            lossvar_eval, lossnino_eval, comloss_eval, sceval
                        )
                    )
                    if sceval > best:
                        torch.save(
                            self.mymodel.state_dict(),
                            chk_path,
                        )
                        best = sceval
                        count = 0
                        print("\nsaving model...")
            # ----------after one epoch-----------
            (
                _,
                _,
                lossvar_eval,
                lossnino_eval,
                comloss_eval,
                sceval,
            ) = self.model_pred(dataloader=dataloader_eval)
            print(
                "\n-->epoch{} end... \nloss_var:{:.3f} \nloss_nino:{:.3f} \nloss_com:{:.3f} \nscore: {:.3f}".format(
                    i_epoch, lossvar_eval, lossnino_eval, comloss_eval, sceval
                )
            )
            if sceval <= best:
                count += 1
                print("\nsc is not increase for {} epoch".format(count))
            else:
                count = 0
                print(
                    "\nsc is increase from {:.3f} to {:.3f}   \nsaving model...\n".format(
                        best, sceval
                    )
                )
                torch.save(
                    self.mymodel.state_dict(),
                    chk_path,
                )
                best = sceval
            # ---------early stop
            if count == self.mypara.patience:
                print(
                    "\n-----!!!early stopping reached, max(sceval)= {:3f}!!!-----".format(
                        best
                    )
                )
                break
        del self.mymodel


if __name__ == "__main__":
    print(mypara.__dict__)
    print("\nloading pre-train dataset...")
    traindataset = make_dataset2(mypara)
    print(traindataset.selectregion())
    print("\nloading evaluation dataset...")
    evaldataset = make_testdataset(
        mypara,
        ngroup=100,
    )
    print(evaldataset.selectregion())
    # -------------------------------------------------------------
    trainer = modelTrainer(mypara)
    trainer.train_model(
        dataset_train=traindataset,
        dataset_eval=evaldataset,
    )
