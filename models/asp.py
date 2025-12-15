# watermark version

import logging
import os
import numpy as np
import torch
from torch import nn
import copy
from torch.serialization import load
from tqdm import tqdm, trange
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from backbone.asp_backbone import SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.data_manager import pil_loader
from utils.utiles import setup_seed, fedavg_aggregation, evaluate_accuracy_forgetting, evaluate_accuracy, train_gen, start, combine_data
from sklearn.metrics import confusion_matrix, roc_auc_score
from copy import deepcopy
from utils import (
    GenDataset,
    DataIter,
    average_weights
)
from omegaconf import OmegaConf
from ldm import DDIMSampler, LatentDiffusion
from einops import rearrange
from PIL import Image
from glob import glob
from torchvision import transforms




# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 1

def cos_loss(cosine, label):
    loss = 0
    for i, y in enumerate(label):
        loss += 1 - cosine[i, y]
    return loss / len(label)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self. batch_size= args["batch_size"]
        self. init_lr=args["init_lr"]
        
        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args
        ####
        self.cur_class = None
        ### diff
        # self.generator_init()
        if args['syn_image_path'] is not None:
            self.syn_imgs_dir = args['syn_image_path']
        else:
            self.syn_imgs_dir = os.path.join(args['save_dir'], "syn_imgs")

    def after_task(self):
        self._known_classes = self._total_classes
        # self._cur_task += 1
    
    def replace_fc(self,trainloader, model, args):
        # use class prototype as classifier weights.
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list=np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            self._network.fc.weight.data[class_index]=proto
        
        return model
    

    def incremental_train(self, teacher, generator, data_manager):
        self._cur_task += 1
        self.generator_init(self._cur_task)

        # import pdb
        # pdb.set_trace()
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", kshot=self.args["kshot"] )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        if isinstance(self.args['kshot'], int) and self._known_classes>0:
            train_bs = self.args['fs_batch_size']
        else:
            train_bs = self.batch_size
        self.train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        test_curr_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test", mode="test" )
        self.test_curr_loader = DataLoader(test_curr_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", kshot=self.args["kshot"])
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        logging.info("training set size: {}, fc construct set size: {}".format(len(train_dataset), len(train_dataset_for_protonet)))
        # import pdb
        # pdb.set_trace()
        # if self._cur_task == data_manager.nb_tasks - 1: # pretrain
        # # if self._cur_task != data_manager.nb_tasks - 1 and self._cur_task != 0:
        #     original_global = deepcopy(self._network)
        #     print(self._total_classes)
        #     classes_learned = self._total_classes
        #     teacher = train_gen(deepcopy(self._network), classes_learned, generator, self.args)
        #     # for client in clients:
        #     #     client.last_valid_dim = classes_learned
        #     #     client.valid_dim = classes_learned + task_size
        #     self._network = original_global
        #     # classes_learned += task_size
        # if self._cur_task == 0:
        #     teacher = None
        teacher = None

        classes_learned = self._total_classes

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        ## diff
        ## train_dataset.images
        ## self.init_dataloader(data_manager)
        gen_dataset = GenDataset(
            input_np_array=train_dataset.images,
            class_ids=train_dataset.labels,
            min_class_id=self._known_classes
        ) # image, caption
        local_gen_dataset = deepcopy(gen_dataset)
        local_gen_loader = DataLoader(local_gen_dataset, batch_size=self.args['g_local_bs'],
                num_workers=4, shuffle=True)
        self.gen_data_iters = []

        self.gen_data_iters.append(DataIter(local_gen_loader))
        self.all_classes = np.unique(train_dataset.labels)
        self.min_class_id, self.max_class_id = np.min(self.all_classes), np.max(self.all_classes)

        if self.args['need_syn_imgs']:
            inv_text_embeds = self._class_inversion() # class inversion for current class
            self._synthesis_imgs(inv_text_embeds)     # data synthesize for current class
        self._init_syn_dataloader()
        # self._fl_train()
        ## diff

        self._train(classes_learned, teacher, self.train_loader, self.test_loader, self.train_loader_for_protonet, self.local_cur_loaders)
        # self._train(classes_learned, teacher, self.train_loader, self.test_loader, self.local_cur_loaders)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self.config['model']["params"]['personalization_config']["params"]['num_classes'] = \
            self.args['increment']
        
        ## task_size = data_manager.get_task_size(self._cur_task)



    def _train(self, classes_learned, teacher, train_loader, test_loader, train_loader_for_protonet, local_cur_loaders):
        
        self._network.to(self._device)

        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info('total parameters: {}'.format(total_params))
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info('trainable parameters: {}'.format(total_trainable_params))

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())
        
        if self._cur_task > 0:
            self.update_ema_prompt(train_loader_for_protonet)  
            self.replace_fc(train_loader_for_protonet, self._network, None)

        # if os.path.exists(self.args["base_model_path"]) and self._cur_task==0: 
        #     logging.info('================= load base model from: {} ================='.format(self.args["base_model_path"]))
        #     self._network.load_state_dict(torch.load(self.args["base_model_path"]))
        # else:
        
        if self.args['optimizer']=='sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer']=='adam':
            optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        
        if teacher is None:
            self._init_train(local_cur_loaders[0], test_loader, optimizer, scheduler)
            # self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._init_train_teacher(classes_learned, teacher, train_loader, test_loader, optimizer, scheduler)

        if self._cur_task == 0:
            torch.save(self._network.state_dict(), self.args["base_model_path"])

        if self._cur_task == 0:
            self.update_ema_prompt(train_loader_for_protonet, mode='base')
            self.replace_fc(train_loader_for_protonet, self._network, None)  

    def eval_task(self):
        y_pred, y_true = self._eval_acc(self.test_loader)
        accy = self._evaluate(y_pred, y_true)
        return accy
    
    def _eval_acc(self, loader):
        self._network.to(self._device)
        self._network.eval()
        y_pred, y_true = [], []
        all_outputs, all_embedding = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                out = self._network(inputs)
                outputs = out["logits"]
                embedding = out["features"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu())
            all_embedding.append(embedding.cpu())
            
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        all_outputs = torch.cat(all_outputs)
        all_embedding = torch.cat(all_embedding)

        return y_pred, y_true # [N, topk]
    
    # naive train
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if isinstance(self.args['kshot'], int) and self._known_classes>0:
            total_epoch = self.args['fs_epoch']
        else:
            total_epoch = self.args['tuned_epoch']
        for _, epoch in enumerate(range(total_epoch)):

            if self._cur_task == 0:
                anchor_samples = self.find_anchor_sample(self._network, self.train_loader_for_protonet)
                print('anchor samples found')
                
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
 
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                if self._cur_task == 0:
                    cur_class = set(targets.cpu())
                    self.cur_class = cur_class
                    for c in cur_class:
                        inputs = torch.cat([inputs, anchor_samples[c].unsqueeze(0).to(self._device)])
                    out = self._network(inputs, self.args["perturb_var"])
                    logits = out["logits"][:-len(cur_class),:]
                    features = out["features"]
                    (mu, std) = out["kl"]
                    sim_loss = 0.0
                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    anchor_id = 0
                    for c in cur_class:
                        fea_c = features[:-len(cur_class)][targets==c]
                        fea_anchor = features[len(cur_class):][anchor_id].detach()
                        fea_anchor = fea_anchor.unsqueeze(0).repeat(len(fea_c), 1)
                        sim_loss += (1-cos(fea_c, fea_anchor)).mean()
                        anchor_id += 1
                    sim_loss = sim_loss / len(cur_class)

                    loss = F.cross_entropy(out["logits"][:-len(cur_class),:], targets) + self.args["anchor_lambda"] * sim_loss 
                    # KL
                    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1) / mu.size(0)
                    loss += self.args["kl_weight"] * KL
                    
                else:
                    logits = self._network(inputs, self.args["perturb_var"])["logits"]
                    logits[:, :self._known_classes] = float('-inf') 
                    loss = F.cross_entropy(logits, targets)
                    _, pre_imgs, pre_labels = self.pre_syn_data_iter.next()
                    pre_imgs, pre_labels = pre_imgs.to(self._device), pre_labels.to(self._device)
                    if self.args["w_ce_pre"] > 0:
                        s_out = self._network(pre_imgs, self.args["perturb_var"])["logits"]
                        logits = s_out["logits"]
                        pre_labels = pre_labels  # 目标标签
                        p = 0.2  # 保留 20%
                        # 提取每个样本对应 pre_labels 的 logits 值
                        batch_size, num_classes = logits.shape
                        pre_logits = logits.gather(1, pre_labels.unsqueeze(1)).squeeze(1)  # [batch_size]
                        # 计算每个样本的阈值（保留 top-p 的 logits）
                        threshold = torch.quantile(pre_logits, 1 - p)  # 计算 pre_logits 的 top-p 阈值
                        # 构造掩码：保留 logits >= 阈值的部分
                        mask = (logits >= threshold).float()  # [batch_size, num_classes]
                        # 应用掩码，将低于阈值的 logits 置零
                        logits_topk = logits * mask
                        # 使用处理后的 logits 计算交叉熵损失
                        loss_ce_pre = F.cross_entropy(logits_topk[:, :self._known_classes], pre_labels)
                        # loss_ce_pre = F.cross_entropy(s_out["logits"][:, :self._known_classes], pre_labels)

                        loss = loss + self.args["w_ce_pre"] * loss_ce_pre

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_cur_acc = self._compute_accuracy(self._network, self.test_curr_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {} => Loss {:.3f}, Train_accy {:.2f}, Test_curr_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                losses / len(train_loader),
                train_acc,
                test_cur_acc,
                test_acc,
            )
            logging.info(info)


    # teacher train
    def _init_train_teacher(self, classes_learned, teacher, train_loader, test_loader, optimizer, scheduler):
        if isinstance(self.args['kshot'], int) and self._known_classes>0:
            total_epoch = self.args['fs_epoch']
        else:
            total_epoch = self.args['tuned_epoch']
        previous_teacher, previous_linear = deepcopy(teacher[0]), deepcopy(teacher[1])

        for _, epoch in enumerate(range(total_epoch)):
            if self._cur_task == 0:
                anchor_samples = self.find_anchor_sample(self._network, self.train_loader_for_protonet)
                print('anchor samples found')

            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            # x_replay, y_replay, y_replay_hat = self.sample(classes_learned, previous_teacher, self.args["syn_size"])       

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # x_replay, y_replay, y_replay_hat = self.sample(classes_learned, previous_teacher, self.args["syn_size"])       
                # inputs, targets = combine_data(((inputs, targets), (x_replay.to(self._device), y_replay.to(self._device))))
                if self._cur_task == 0:
                    cur_class = set(targets.cpu())
                    self.cur_class = cur_class
                    for c in cur_class:
                        inputs = torch.cat([inputs, anchor_samples[c].unsqueeze(0).to(self._device)])
                    out = self._network(inputs, self.args["perturb_var"])
                    logits = out["logits"][:-len(cur_class),:]
                    features = out["features"]
                    (mu, std) = out["kl"]
                    sim_loss = 0.0
                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    anchor_id = 0
                    for c in cur_class:
                        fea_c = features[:-len(cur_class)][targets==c]
                        fea_anchor = features[len(cur_class):][anchor_id].detach()
                        fea_anchor = fea_anchor.unsqueeze(0).repeat(len(fea_c), 1)
                        sim_loss += (1-cos(fea_c, fea_anchor)).mean()
                        anchor_id += 1
                    sim_loss = sim_loss / len(cur_class)

                    loss = F.cross_entropy(logits, targets) + self.args["anchor_lambda"] * sim_loss 
                    # KL
                    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1) / mu.size(0)
                    loss += self.args["kl_weight"] * KL
                    
                else:
                    logits = self._network(inputs, self.args["perturb_var"])["logits"]
                    logits[:, :self._known_classes] = float('-inf') 
                    loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_cur_acc = self._compute_accuracy(self._network, self.test_curr_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {} => Loss {:.3f}, Train_accy {:.2f}, Test_curr_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                losses / len(train_loader),
                train_acc,
                test_cur_acc,
                test_acc,
            )
            logging.info(info)

    def sample(self, classes_learned, teacher, size, return_scores=True):
        return teacher.sample(size, classes_learned, return_scores=return_scores)

    def update_ema_prompt(self, train_loader, mode='new'):
        self._network.eval()
        prompt_list = []

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                prompt, _ = self._network.backbone.Prompt_Encoder(data, self._network.backbone.TIP, 0)
                prompt_list.append(prompt.detach().cpu())

        if mode == 'new':
            self._network.backbone.Avg_TSP = self.args["EMA_beta"]*self._network.backbone.Avg_TSP + (1-self.args["EMA_beta"])*torch.mean(torch.cat(prompt_list, dim=0), dim=0) 
        else:
            self._network.backbone.Avg_TSP = torch.mean(torch.cat(prompt_list, dim=0), dim=0) 

        self._network.backbone.Avg_TSP.to(self._device)   



    def find_anchor_sample(self, model, train_loader):
        # train_loader must be Shuffle == False.

        model.eval()
        embedding_list = []
        label_list = []
        prompt_list = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

                prompt, _ = self._network.backbone.Prompt_Encoder(data, self._network.backbone.TIP, 0)
                prompt_list.append(prompt.detach().cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        self._network.backbone.Avg_TSP = torch.mean(torch.cat(prompt_list, dim=0), dim=0)   
        self._network.backbone.Avg_TSP.to(self._device)   

        class_list=np.unique(train_loader.dataset.labels)
        anchor_sample = []
        for class_index in class_list:
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            class_mean = embedding.mean(0)
            class_mean = class_mean.unsqueeze(0).repeat(len(embedding), 1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_sim = cos(embedding, class_mean)
            anchor_index = torch.argmax(cos_sim)
            anchor_sample.append(train_loader.dataset[data_index[anchor_index]][1])
        return anchor_sample
    

    def _class_inversion(self):
        self._generator.to(self._device)
        ############
        # self._generator.embedding_manager.load_state_dict(self.generator_init_embedding)
        ############
        prog_bar = tqdm(range(self.args["com_round_gen"]), desc='Class Inversion')
        # for _ in prog_bar:
        local_weights = []
        # m = max(int(self.args["frac"] * self.args["num_users"]), 1)
        # idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
        # for idx in idxs_users:
        idx = 0
        # if self.gen_data_iters[idx] is None:
        #     continue
        w = self._local_update_g(deepcopy(self._generator), self.gen_data_iters[idx])
        # w = self._local_update_g(self._generator,
        #                         self.gen_data_iters[idx])
        local_weights.append(deepcopy(w))
        global_weights = average_weights(local_weights, self.args['g_sigma'])
        self._generator.embedding_manager.load_state_dict(global_weights)
        inv_text_embeds = deepcopy(self._generator.embedding_manager.string_to_param_dict)
        if self.args["save_cls_embeds"]:
            cls_embeds_path = os.path.join(self.save_dir, 
                'cls_embeds_ckpt', '%d-%d_embedding_manager.pt' % (self.min_class_id, self.max_class_id))
            os.makedirs(os.path.dirname(cls_embeds_path), exist_ok=True)
            torch.save(self._generator.embedding_manager.state_dict(), cls_embeds_path)
        return inv_text_embeds

    def generator_init(self, cur_task):
        self.config = OmegaConf.load(self.args['config'])
        self.config.model.params.ckpt_path = self.args['ldm_ckpt']
        if cur_task == 0:
            self.config['model']["params"]['personalization_config']["params"]['num_classes'] = \
                self.args['init_cls']
        else:
            self.config['model']["params"]['personalization_config']["params"]['num_classes'] = \
                self.args['increment']            
        self._generator = LatentDiffusion(**self.config['model']["params"])
        self._generator.load_state_dict(
            torch.load(self.args['ldm_ckpt'], map_location="cpu")["state_dict"], 
            strict=False)
        self.generator_init_embedding = deepcopy(self._generator.embedding_manager.state_dict())

        self._generator.learning_rate =  self.config.data.params.batch_size * self.config.model.base_learning_rate
        print("Setting learning rate to {:.2e} =  {} (batchsize) * {:.2e} (base_lr)".format(
                self._generator.learning_rate, 
                self.config.data.params.batch_size, 
                self.config.model.base_learning_rate))

    def _local_update_g(self, generator, gen_data_loader):
        generator.train()
        generator = generator.to(self._device)
        optim = generator.configure_optimizers()
        for _ in range(self.args["g_local_train_steps"]):
            batch = gen_data_loader.next()
            batch["image"] = batch["image"].to(self._device)
            loss, _ = generator.shared_step(batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
        return generator.embedding_manager.state_dict()
    
    def _synthesis_imgs(self, inv_text_embeds):
        self._generator.embedding_manager.string_to_param_dict = inv_text_embeds
        sampler = DDIMSampler(self._generator)
        # outdir = os.path.join('gen_result_tmp', str(idx))
        outdir = os.path.join(self.syn_imgs_dir, "task_{}".format(self._cur_task))
        os.makedirs(outdir, exist_ok=True)
        prompt = "a photo of *"
        n_samples = 40
        scale = 10.0
        ddim_steps = 50
        ddim_eta = 0.0
        H = 256
        W = 256
        with torch.no_grad():
            for tmp_cls in self.all_classes:
                base_count = 0
                with self._generator.ema_scope():
                    uc = None
                    tmp_cls_tensor = torch.LongTensor([tmp_cls - self.min_class_id,] * n_samples)
                    if scale != 1.0:
                        uc = self._generator.get_learned_conditioning(n_samples * [""], tmp_cls_tensor)
                    for _ in trange(self.args['n_iter'], desc="Sampling"):
                        c = self._generator.get_learned_conditioning(n_samples * [prompt], tmp_cls_tensor)
                        shape = [4, H//8, W//8]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                        conditioning=c,
                                                        batch_size=n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=uc,
                                                        eta=ddim_eta)
                        x_samples_ddim = self._generator.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            if not os.path.exists(os.path.join(outdir, str(tmp_cls))):
                                os.makedirs(os.path.join(outdir, str(tmp_cls)))
                            Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(outdir, str(tmp_cls), f"{tmp_cls}-{base_count}.jpg"))
                            base_count += 1

    def _init_syn_dataloader(self):
        cur_syn_dataset = TaskSynImageDataset(
            self.syn_imgs_dir, self._cur_task, self.args['cur_size'],
            transform=train_transform[self.args['dataset_syn']])  ## 这个地方需要注意、是否和baseline匹配
        self.local_cur_loaders = []
        if self._cur_task > 0:
            batch_size = self.args["local_bs1"]
        else:
            batch_size = self.args["local_bs0"]
        for idx in range(self.args["num_users"]):
            self.local_cur_loaders.append(
                DataLoader(ConcatDataset([self.train_dataset, cur_syn_dataset]), 
                           batch_size=batch_size, shuffle=True, num_workers=4))
        if self._cur_task >= 0: 
            pre_syn_dataset = ConcatDataset(
                [TaskSynImageDataset(self.syn_imgs_dir, i, self.args['pre_size'],
                                     transform=train_transform[self.args['dataset_syn']]) 
                 for i in range(self._cur_task)])
            pre_syn_data_loader = DataLoader(
                pre_syn_dataset, batch_size=128, shuffle=True,
                num_workers=4, pin_memory=True)
            self.pre_syn_data_iter = DataIter(pre_syn_data_loader)

class TaskSynImageDataset(Dataset):
    def __init__(self, root, task_id, size_per_cls, transform=None):
        self.root = os.path.join(root, "task_{}".format(task_id))
        img_paths = glob(os.path.join(self.root, "*", "*.jpg"))
        img_paths = [tp for tp in img_paths 
                     if int(tp.split("-")[-1].rstrip(".jpg")) < size_per_cls]
        self.images, self.labels = [], []
        for tp in img_paths:
            self.labels.append(int(tp.split("/")[-2]))
            with Image.open(tp) as tmp_img:
                self.images.append(np.array(tmp_img))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return idx, img, label

    def __len__(self):
        return len(self.images)
    
train_transform = {
    "cifar100": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]),
    "tiny_imagenet": transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}