import torch
import faiss
from torch import nn
from models import SAINT

from data_openml import data_prep_openml, DataSetCatCon, data_prep_openml_credit
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import classification_scores_rac
from augmentations import embed_data_mask
import os
import numpy as np

parser = argparse.ArgumentParser()


class MLP(nn.Module):
    def __init__(self, d_main, d_multiplier, n_classes, norm=False, residual=True):
        super().__init__()
        self.norm = norm
        if self.norm:
            self.norml = nn.LayerNorm(d_main)
        self.mlp1 = nn.Linear(d_main, d_main * d_multiplier)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.2)
        self.mlp2 = nn.Linear(d_main * d_multiplier, n_classes)
        self.residual = residual
        if d_main != n_classes:
            self.residual = False

    def forward(self, inp):
        x = inp
        if self.norm:
            x = self.norml(x)
        x = self.mlp1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.mlp2(x)
        if self.residual:
            x = inp + x
        return x


class RACModel(nn.Module):
    def __init__(self, encoder, candidate_embeddings, candidate_y, search_index, n_classes, d_main, d_multiplier,
                 context_size, Abl=-1, num_pred_blocks=1):
        super().__init__()
        self.encoder = encoder
        self.candidate_embeddings = candidate_embeddings
        self.candidate_y = candidate_y
        self.search_index = search_index
        self.label_encoder = nn.Embedding(n_classes, d_main)
        print(f"Context size is: {context_size}")
        self.context_size = context_size
        self.key_proj = MLP(d_main, 1, d_main, True)
        self.predictor = nn.ModuleList(
            [MLP(d_main * d_multiplier ** i, d_multiplier, d_main * d_multiplier ** (i + 1)) for i in
             range(num_pred_blocks - 1)])
        self.predictor += [MLP(d_main * d_multiplier ** num_pred_blocks, d_multiplier, n_classes)]
        self.dropout = nn.Dropout(0.2)
        self.Abl = Abl

    def forward(self, x_cat, x_cont, is_train, candidate_embeddings=None, candidate_y=None,
                search_index=None, context_freeze=True):
        if not context_freeze:
            if candidate_embeddings is not None:
                self.candidate_embeddings = candidate_embeddings
            if candidate_y is not None:
                self.candidate_y = candidate_y
            if search_index is not None:
                self.search_index = search_index
            reps = self.encoder.transformer(x_cat, x_cont)
        else:
            with torch.no_grad():
                reps = self.encoder.transformer(x_cat, x_cont)

        k = reps[:, 0, :]
        distances, context_idx = self.search_index.search(k.clone().detach().cpu().numpy(),
                                                          self.context_size + (1 if is_train else 0))
        distances = torch.from_numpy(distances)
        context_idx = torch.from_numpy(context_idx)
        if is_train:
            context_idx = context_idx.gather(-1, distances.argsort()[:, 1:])

        context_k = self.candidate_embeddings[context_idx].clone()
        context_y = self.candidate_y[context_idx].clone()
        context_k = context_k.to(k.device)
        context_y = context_y.to(k.device)
        similarities = (
                -k.square().sum(-1, keepdim=True)
                + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
                - context_k.square().sum(-1)
        )
        probs = torch.nn.functional.softmax(similarities, dim=-1)
        probs = self.dropout(probs)
        context_y_emb = self.label_encoder(context_y.squeeze())

        if self.Abl == 1:
            values = self.key_proj(context_k)  # Ablation 1
        elif self.Abl == 2:
            values = self.key_proj(k[:, None] - context_k)  # Ablation 2
        elif self.Abl == 2:
            values = context_y_emb + self.key_proj(context_k)  # Ablation 3
        else:
            values = context_y_emb + self.key_proj(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = k + context_x
        y = self.predictor(x)
        return y


def computeEmbd(encoder, trainloader, device, vision_dset):
    encoder.eval()
    search_index = faiss.IndexFlatL2(opt.embedding_size)
    search_index.reset()
    index_wpr = faiss.IndexIDMap(search_index)
    embds = []
    ys = []
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), \
            data[3].to(device), data[4].to(device)
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, encoder, vision_dset)
            reps = encoder.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:, 0, :]

            embds.append(y_reps.cpu())
            ys.append(y_gts.cpu())
            st = i * trainloader.batch_size
            ed = st + y_reps.size(0)
            idx = np.arange(st, ed).astype('int64')
            index_wpr.add_with_ids(y_reps.cpu().numpy().astype('float32'), idx)
    embds = torch.concat(embds, 0)
    ys = torch.concat(ys, 0)

    return embds, ys, index_wpr


parser.add_argument('--dset_id', required=True, type=int)
parser.add_argument('--vision_dset', action='store_true')
parser.add_argument('--task', required=True, type=str, choices=['binary', 'multiclass', 'regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,
                    choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])

parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--rac_exp_name', default='rac', type=str)
parser.add_argument('--set_seed', default=1, type=int)
parser.add_argument('--dset_seed', default=5, type=int)
parser.add_argument('--active_log', action='store_true')
parser.add_argument('--split_type', default='In Time', type=str, choices=['In Time', 'Out Of Time'])

parser.add_argument('--Abl', default=-1, type=int)
parser.add_argument('--num_pred_blocks', default=1, type=int)
parser.add_argument('--context_size', default=100, type=int)
parser.add_argument('--context_freeze_epoch', default=-1, type=int)
parser.add_argument('--encoder_type', default='finetuned', type=str, choices=['pretrained', 'finetuned'])
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*',
                    choices=['contrastive', 'contrastive_sim', 'denoising'])
parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_mask_prob', default=0, type=float)
parser.add_argument('--mask_prob', default=0, type=float)

parser.add_argument('--ssl_avail_y', default=0, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])

opt = parser.parse_args()
modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, str(opt.dset_id), opt.run_name)
rac_modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, str(opt.dset_id), opt.run_name,
                                  opt.rac_exp_name)
embedding_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, str(opt.dset_id), opt.run_name, 'embedding')

if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)
os.makedirs(rac_modelsave_path, exist_ok=True)
os.makedirs(embedding_path, exist_ok=True)

if opt.active_log:
    import wandb

    if opt.pretrain:
        wandb.init(project="saint_v2_all", group=opt.run_name,
                   name=f'pretrain_{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
    else:
        if opt.task == 'multiclass':
            wandb.init(project="saint_v2_all_kamal", group=opt.run_name,
                       name=f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
        else:
            wandb.init(project="saint_v2_all", group=opt.run_name,
                       name=f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')

print('Downloading and processing the dataset, it might take some time.')

if opt.split_type == 'Out Of Time':
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml_credit(
        opt.dset_id, opt.dset_seed, opt.task, datasplit=[.65, .15, .2])

else:
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml(
        opt.dset_id, opt.dset_seed, opt.task)

continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

##### Setting some hyperparams based on inputs and dataset
_, nfeat = X_train['data'].shape
if nfeat > 100:
    opt.embedding_size = min(8, opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = min(4, opt.attention_heads)
    opt.attention_dropout = 0.8
    opt.embedding_size = min(32, opt.embedding_size)
    opt.ff_dropout = 0.8

print(nfeat, opt.batchsize)
print(opt)

if opt.active_log:
    wandb.config.update(opt)

train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)

valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, opt.dtask, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)

test_ds = DataSetCatCon(X_test, y_test, cat_idxs, opt.dtask, continuous_mean_std)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)

if opt.task == 'regression':
    y_dim = 1
else:
    y_dim = len(np.unique(y_train['data'][:, 0]))

cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(
    int)

encoder_model = SAINT(
    categories=tuple(cat_dims),
    num_continuous=len(con_idxs),
    dim=opt.embedding_size,
    dim_out=1,
    depth=opt.transformer_depth,
    heads=opt.attention_heads,
    attn_dropout=opt.attention_dropout,
    ff_dropout=opt.ff_dropout,
    mlp_hidden_mults=(4, 2),
    cont_embeddings=opt.cont_embeddings,
    attentiontype=opt.attentiontype,
    final_mlp_style=opt.final_mlp_style,
    y_dim=y_dim
)
vision_dset = opt.vision_dset

if y_dim == 2 and opt.task == 'binary':
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and opt.task == 'multiclass':
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
else:
    raise 'case not written yet'

encoder_model.to(device)

if opt.encoder_type == 'pretrained':
    print("Loading only the pretrained SAINT model")
    encoder_model.load_state_dict(torch.load('%s/pretrainedModel.pth' % (modelsave_path)))

else:
    print("Loading only the finetuned SAINT model")
    encoder_model.load_state_dict(torch.load('%s/bestmodel.pth' % (modelsave_path)))

encoder_model.eval()

with torch.no_grad():
    embds, ys, index_wpr = computeEmbd(encoder_model, trainloader, device, opt.task, vision_dset)
    torch.save({'embds': embds, 'y': ys}, f"{embedding_path}/embedding.torch")
    faiss.write_index(index_wpr, f"{embedding_path}/faiss.index")
encoder_model.train()

model = RACModel(
    encoder_model,
    embds,
    ys,
    index_wpr,
    y_dim,
    opt.embedding_size,
    1,
    opt.context_size,
    opt.Abl,
    opt.num_pred_blocks
)
model.to(device)

if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler

    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

best_valid_auroc = 0
best_valid_aucpr = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_aucpr = 0
best_test_accuracy = 0
best_valid_rmse = 100000
print('Training begins now.')

context_freeze = False
for epoch in range(opt.epochs):
    if opt.context_freeze_epoch >= 0 and epoch >= opt.context_freeze_epoch:
        print("Freezing Context")
        context_freeze = True
        embds, ys, index_wpr = None, None, None
    else:
        context_freeze = False
        embds, ys, index_wpr = computeEmbd(model.encoder, trainloader, device, opt.task, vision_dset)
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[
            3].to(device), data[4].to(device)
        _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model.encoder, vision_dset)
        y_outs = model(x_categ_enc, x_cont_enc, True, embds, ys, index_wpr, context_freeze)
        if opt.task == 'regression':
            loss = criterion(y_outs, y_gts)
        else:
            loss = criterion(y_outs, y_gts.squeeze())
        loss.backward()
        optimizer.step()
        if opt.optimizer == 'SGD':
            scheduler.step()
        running_loss += loss.item()
    print(running_loss)
    if epoch % 1 == 0:
        model.eval()
        with torch.no_grad():
            if opt.task in ['binary', 'multiclass']:
                accuracy, auroc, aucpr = classification_scores_rac(model.encoder, validloader, device, opt.task, vision_dset)
                test_accuracy, test_auroc, test_aucpr = classification_scores_rac(model.encoder, testloader, device, opt.task,
                                                                                  vision_dset)

                print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f, VALID AUCPR: %.3f' %
                      (epoch + 1, accuracy, auroc, aucpr))
                print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f, TEST AUCPR: %.3f' %
                      (epoch + 1, test_accuracy, test_auroc, test_aucpr))
                if opt.task == 'multiclass':
                    if accuracy > best_valid_accuracy:
                        best_valid_accuracy = accuracy
                        best_test_auroc = test_auroc
                        best_test_aucpr = test_aucpr
                        best_test_accuracy = test_accuracy
                        torch.save(model.state_dict(), '%s/bestmodel.pth' % (rac_modelsave_path))
                else:
                    if accuracy > best_valid_accuracy:
                        best_valid_aucpr = aucpr
                        best_valid_auroc = auroc
                        best_test_auroc = test_auroc
                        best_test_aucpr = test_aucpr
                        best_test_accuracy = test_accuracy
                        torch.save(model.state_dict(), '%s/bestmodel.pth' % (rac_modelsave_path))

        model.train()

if opt.task == 'binary':
    print('AUROC on best model:  %.3f' % (best_test_auroc))
    print('AUCPR on best model:  %.3f' % (best_test_aucpr))
elif opt.task == 'multiclass':
    print('Accuracy on best model:  %.3f' % (best_test_accuracy))
