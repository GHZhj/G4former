import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import (
    f1_score, roc_auc_score,matthews_corrcoef,
    average_precision_score,
)
from scipy.stats import pearsonr
from tqdm import tqdm
import torch.optim as optim
import os
from transformers import AutoModel

from utils.tokenizersM import get_ntv3_tokenizer
import math
import torch.nn.functional as F
import random

warnings.filterwarnings('ignore')

# Ensure reproducibility: make results consistent across runs
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)

def _read_signal_file(file, m=False):
    signals = [] 
    with open(file) as f:

        for line in f:
            parts = line.strip().split()
            if m:
                signal_values = [x * 0.01 for x in map(float, parts[4:])]
            else:
                signal_values = list(map(float, parts[4:]))  # 从第5列开始读取200个值

            signals.append(signal_values)

    signals = torch.tensor(signals, dtype=torch.float32)
    return  signals  

def train_val(model, train_loader, val_loader, criterion, device, optimizer, scheduler,
              epochs, patience ):
    best_val_loss = float('inf')
    best_val_auprc = 0.0
    best_model_wts = None
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # 记录最优验证集结果
    best_val_preds, best_val_targets, best_val_probs = None, None, None
    
    species_token = 27      # species = 'human'

    for epoch in range(epochs):
        # === TRAIN ===
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for input_ids, w, y in loop:
            input_ids, w, y = input_ids.to(device),w.to(device), y.to(device)

            species_ids = torch.tensor([species_token] * len(y), dtype=torch.long).to(device)  # batch_size

            optimizer.zero_grad()
            out= model(input_ids, species_ids, w)
            loss = criterion(out, y)  

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            preds = out.argmax(1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            loop.set_postfix({"Train Loss": f"{loss:.4f}"})

        train_losses.append(train_loss / train_total)
        train_accs.append(train_correct / train_total)

        # === VALIDATION ===
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds, val_targets, val_probs = [], [], []

        loop = tqdm(val_loader, desc=f"Val | Epoch {epoch+1} [Val]")
        with torch.no_grad():
            for input_ids, w, y in loop:
                input_ids,  w, y = input_ids.to(device),  w.to(device), y.to(device)
                species_ids = torch.tensor([species_token] * len(y), dtype=torch.long).to(device)  # batch_size                

                out= model(input_ids, species_ids, w)
                loss = criterion(out, y)  
                    
                loop.set_postfix({"Val Loss": f"{loss.item():.4f}"})

                val_loss += loss.item() * y.size(0)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                pred_labels = out.argmax(1).cpu().numpy()
                val_preds.extend(pred_labels)
                val_targets.extend(y.cpu().numpy())
                val_probs.extend(probs)
                val_correct += (pred_labels == y.cpu().numpy()).sum()
                val_total += y.size(0)

        val_losses.append(val_loss / val_total)
        val_accs.append(val_correct / val_total)

        # === Calculate AUPRC for this epoch ===
        try:
            current_auprc = average_precision_score(val_targets, val_probs)
        except Exception:
            current_auprc = float('nan')

        scheduler.step()

        # === Early Stopping & Best Model Tracking (by AUPRC) ===
        if current_auprc > best_val_auprc:
            best_val_auprc = current_auprc
            best_model_wts = model.state_dict()
            best_epoch = epoch
            patience_counter = 0
            # ✅ 保存该轮验证集的预测结果
            best_val_preds = val_preds.copy()
            best_val_targets = val_targets.copy()
            best_val_probs = val_probs.copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

        print(f"[Epoch {epoch+1}] "
              f"Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, "
              f"Train Acc={train_accs[-1]:.4f}, Val Acc={val_accs[-1]:.4f}, "
              f"AUPRC={current_auprc:.4f}, Best AUPRC={best_val_auprc:.4f}")
        # === SAVE MODEL EVERY EPOCH ===
        if name == "293T":
            epoch_model_path = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/ck/results/save_L1k_-11-56_293T_epoch_1_64_{lr}_{name}.pth'
        else:
            epoch_model_path = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/ck/results/save_L1k_-11-56_epoch_1_64_{lr}_{name}.pth'

        torch.save(model.state_dict(), epoch_model_path)
        print(f"💾 Saved model for epoch {epoch+1} to {epoch_model_path}")

    # === SAVE BEST MODEL ===
    if best_model_wts is not None:
        if name == "293T":
            model_path = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/ck/results/save_L1k_-11-56_293T_{batch_size}_{lr}_{name}.pth'
        else:
            model_path = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/ck/results/save_L1k_-11-56_{batch_size}_{lr}_{name}.pth'

        torch.save(best_model_wts, model_path)
        model.load_state_dict(best_model_wts)
        print(f"✅ Saved best model to {model_path}")

    # === COMPUTE BEST METRICS ===
    if best_val_targets is not None:
        val_df = pd.DataFrame({
            'true': best_val_targets,
            'pred': best_val_preds,
            'prob': best_val_probs
        })
        csv_path = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/ck/results/save_L1k_-11-56_{batch_size}_{lr}_{name}_best_preds.csv'
        val_df.to_csv(csv_path, index=False)
        print(f"📁 Saved best validation predictions to {csv_path}")

        # === 计算指标 ===
        pearson_corr, _ = pearsonr(best_val_targets, best_val_probs)
        f1 = f1_score(best_val_targets, best_val_preds)
        try:
            auc = roc_auc_score(best_val_targets, best_val_probs)
        except:
            auc = float('nan')
        try:
            auprc = average_precision_score(best_val_targets, best_val_probs)
        except:
            auprc = float('nan')
        mcc = matthews_corrcoef(best_val_targets, best_val_preds)

        print(f"📊 ✅ [Best @ Epoch {best_epoch+1}] "
              f"F1={f1:.4f}, AUC={auc:.4f}, AUPRC={auprc:.4f}, MCC={mcc:.4f}, r={pearson_corr:.4f}")

    return f1, auc, auprc, mcc, val_losses, train_accs, val_accs

class GenomicDataset(Dataset):
    def __init__(self, fasta_pos_fu, methylation_pos_fusame,  fasta_pos_z, methylation_pos_zsame,  
    fasta_neg_fu, methylation_neg_fusame, fasta_neg_z, methylation_neg_zsame, 
    Dnase_Z_F, Dnase_Z_Z, Dnase_F_F, Dnase_F_Z 
   
    ):
        TATAL = 1024
        length = 1024
        start_index = (TATAL - length) // 2
        end_index = start_index + length
        print(start_index)
 
        # pos_seqs_fu, pos_chr_fu = self.read_fasta(fasta_pos_fu)
        # pos_seqs_z, pos_chr_z = self.read_fasta(fasta_pos_z)
        # neg_seqs_fu, neg_chr_fu = self.read_fasta(fasta_neg_fu)
        # neg_seqs_z, neg_chr_z = self.read_fasta(fasta_neg_z)
        # self.fasta_pos = pos_seqs_fu + pos_seqs_z
        # self.fasta_neg = neg_seqs_fu + neg_seqs_z
        # self.chr_pos = pos_chr_fu + pos_chr_z
        # self.chr_neg = neg_chr_fu + neg_chr_z
        # sequences = self.fasta_pos + self.fasta_neg
 
        # # --- Chromosome-based splitting ---
        # # Collect chromosome IDs for all samples
        # all_chr = self.chr_pos + self.chr_neg
        # num_samples = len(all_chr)
        # # Indices for pos and neg
        # pos_indices = list(range(len(self.fasta_pos)))
        # neg_indices = list(range(len(self.fasta_pos), len(self.fasta_pos) + len(self.fasta_neg)))
        # # Determine test set: all chr1
        # test_idx = [i for i, c in enumerate(all_chr) if c == "chr1"]
        # # Remaining indices for train/val
        # remain_idx = [i for i in range(num_samples) if i not in test_idx]
        # # Shuffle remaining indices for random split
        # rng = np.random.RandomState(42)
        # remain_idx_shuffled = remain_idx.copy()
        # rng.shuffle(remain_idx_shuffled)
        # n_train = int(0.8 * len(remain_idx_shuffled))
        # train_idx = remain_idx_shuffled[:n_train]
        # val_idx = remain_idx_shuffled[n_train:]
        # # Save split indices
        # idx_save_path = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/A549/processed/2k_L1k_k_{name}_split_indices.pt'
        # if not os.path.exists(idx_save_path):
        #     split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
        #     torch.save(split_indices, idx_save_path)
        #     print(f"Saved split indices to {idx_save_path}")

        idx_save_path = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/A549/processed/k1_L1k_k_{name}_NT.pt'
        if not os.path.exists(idx_save_path):
            self.fasta_pos = self.read_fasta(fasta_pos_fu) + self.read_fasta(fasta_pos_z) 
            self.fasta_neg = self.read_fasta(fasta_neg_fu) + self.read_fasta(fasta_neg_z) 
            sequences = self.fasta_pos + self.fasta_neg
            # 初始化 tokenizer
            tokenizer = get_ntv3_tokenizer()

            # tokenizer 输出 numpy array
            tokens_np = tokenizer.batch_np_tokenize(sequences)  # shape: [batch, seq_len]
            self.input_ids = torch.tensor(tokens_np, dtype=torch.long)
            torch.save(self.input_ids, idx_save_path)
            self.input_ids = torch.load(idx_save_path) [:, start_index:end_index,]  #.numpy()

            print(f"Saved split indices to {idx_save_path}")
        else:
            self.input_ids = torch.load(idx_save_path) [:, start_index:end_index,]  #.numpy()


        # Dnase_Z_F = _read_signal_file(Dnase_Z_F)
        # Dnase_Z_Z = _read_signal_file(Dnase_Z_Z)
        # Dnase_F_F = _read_signal_file(Dnase_F_F)
        # Dnase_F_Z = _read_signal_file(Dnase_F_Z)
        # self.Dnase = np.concatenate([
        #                     torch.flip(Dnase_Z_F, dims=[1])  , 
        #                     Dnase_Z_Z , 
        #                     torch.flip(Dnase_F_F, dims=[1]) , 
        #                     Dnase_F_Z, 
        #                     ], axis=0)

        # self.labels = np.array([1] * len(self.fasta_pos) + [0] * len(self.fasta_neg))

        # # Save preprocessed features if not exist
        labels_path = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/A549/processed/L1k_1_k_{name}_labels_S.pt'
        if os.path.exists(labels_path):
            print(f"{labels_path} already exists. Loading labels...")
            self.labels = torch.load(labels_path)#.numpy()
            # self.labels2 = torch.load(f'/home/hjzhang/dataset/home-1/ylxiong/Center/Hela_input/processed/Hela_labels.pt')#.numpy()
            # self.labels = torch.cat([ self.labels , self.labels2],0) # 添加首尾

        else:
            torch.save(torch.tensor(self.labels, dtype=torch.int32), labels_path)
            print(f"Saved labels to {labels_path}")

        # methylation_pos_fudif = _read_signal_file(methylation_pos_fusame  ,m=True)
        # methylation_pos_zdif = _read_signal_file(methylation_pos_zsame ,m=True)
        # methylation_neg_fudif = _read_signal_file(methylation_neg_fusame ,m=True)
        # methylation_neg_zdif = _read_signal_file(methylation_neg_zsame ,m=True)

        # self.methylation = np.concatenate([
        #                     torch.flip(methylation_pos_fudif, dims=[1])  , 
        #                     methylation_pos_zdif , 
        #                     torch.flip(methylation_neg_fudif, dims=[1]) , 
        #                     methylation_neg_zdif , 
        #                     ], axis=0)
             

        epi_path = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/A549/processed/L1k_1_k_{name}_epis_WGBS_3_S.pt'
        if os.path.exists(epi_path):
            print(f"{epi_path} already exists. Loading compressed features...")
            self.epi = torch.load(epi_path).float() 
            # self.epi2 = torch.load(f'/home/hjzhang/dataset/home-1/ylxiong/Center/Hela_input/processed/Hela_epis_WGBS.pt').float()
            # self.epi = torch.cat([ self.epi , self.epi2],0) # 添加首尾
            print(self.epi.max())

        else:
            print("Computing and saving compressed epi features...")
            epi_tensor = torch.tensor( self.methylation[:, :, np.newaxis]  , dtype=torch.float32)
            torch.save(epi_tensor, epi_path)
            self.epi = epi_tensor.float()
            print(f"Saved compressed epi features to {epi_path}")

        Dnase_path = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/A549/processed/L1k_1_k_{name}_epis_dnase_S.pt'
        if os.path.exists(Dnase_path):
            print(f"{Dnase_path} already exists. Loading compressed features...")
            self.Dnase = torch.log1p(torch.load(Dnase_path).float())
            print(self.Dnase.max())

        else:
            print("Computing and saving compressed epi features...")
            epi_tensor = torch.tensor( self.Dnase[:, :, np.newaxis]  , dtype=torch.float32)
            torch.save(epi_tensor, Dnase_path)
            self.Dnase = torch.log1p( epi_tensor.float())
            print(f"Saved compressed epi features to {Dnase_path}")

    def read_fasta(self, file):
        seqs = []
        with open(file) as f:
            seq = ''
            for line in f:
                if line.startswith('>'):
                    if seq:
                        seqs.append(seq.upper())
                        seq = ''
                else:
                    seq += line.strip()
            if seq:
                seqs.append(seq.upper())
        return seqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        train_seq = torch.cat([ self.Dnase[idx],  self.epi[idx]],-1) # 添加首尾

        return (
            self.input_ids[idx] , train_seq ,
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

class AdaptiveLayerNorm(nn.LayerNorm):
    """LayerNorm that applies per-condition affine modulation."""

    def __init__(
        self, num_features: int, conditions_dims: list[int], epsilon: float = 1e-5
    ):
        super().__init__(
            normalized_shape=num_features, eps=epsilon, elementwise_affine=True
        )
        self.modulation_layers = nn.ModuleList(
            [nn.Linear(cd, 2 * num_features) for cd in conditions_dims]
        )
        self._num_conditions = len(conditions_dims)
        self._dim = num_features

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self._base_ln_fp32(x)

        if len(conditions) != self._num_conditions:
            raise ValueError("Number of conditions mismatch")

        if conditions_masks is None:
            conditions_masks = [
                torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
                for _ in conditions
            ]

        scale = torch.ones_like(x[:, :1, :])
        shift = torch.zeros_like(x[:, :1, :])

        for i, cond in enumerate(conditions):
            cond_cast = cond.to(self.modulation_layers[i].weight.dtype)
            tmp = self.modulation_layers[i](cond_cast).unsqueeze(1)
            tmp = tmp.to(x.dtype)
            shift_i, scale_i = torch.chunk(tmp, 2, dim=-1)
            mask = conditions_masks[i].unsqueeze(-1).unsqueeze(-1)
            shift_i = torch.where(mask.bool(), shift_i, 0.0)
            scale_i = torch.where(mask.bool(), scale_i, 0.0)
            scale = scale * (1.0 + scale_i)
            shift = shift + shift_i

        return x * scale + shift

    def _base_ln_fp32(self, x: torch.Tensor) -> torch.Tensor:
        """Run base LayerNorm in fp32 (compiler-friendly, like Mistral/Gemma)."""
        # Compute in fp32
        x_fp32 = x.to(torch.float32)
        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = ((x_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
        x_normed = (x_fp32 - mean) * torch.rsqrt(var + self.eps)
        
        # Apply inherited weight/bias in fp32, then cast back
        if self.weight is not None:
            x_normed = x_normed * self.weight.to(torch.float32)
        if self.bias is not None:
            x_normed = x_normed + self.bias.to(torch.float32)
        
        return x_normed.type_as(x)

class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd):
        super().__init__()
        n_head = 12
        block_size = 1024
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

        self.n_head = n_head        

    def forward(self, q, k, v, layer_past=None, condition=None, mask=True):
        B, T, C = q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(k).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(q).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(v).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x, condition=None, mask=True):
        x = self.ln1(x)
        y, attn = self.attn(x, x, x, condition=condition, mask=mask)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

class G4former(nn.Module):
    def __init__(self):
        super(G4former, self).__init__()     

        self.adaptive_norm = AdaptiveLayerNorm(
            num_features=768, 
            conditions_dims=[ 1024 ]  # w 的条件维度，例如2或更多
        )
        # self.fc2 = nn.Linear(1, 768)  # Second fully connected layer (optional)
        self.hidden_dim = 768

        self.classifier = nn.Linear(self.hidden_dim, 2)  # Binary classification
        self.activation = nn.GELU()  # Activation function

        self.pytorch_model = AutoModel.from_pretrained(
                "/home/hjzhang/dataset/home-1/ylxiong/caduceus/nucleotide_transformer_v3/ntv3_100m_post",
                    trust_remote_code=True
                )

        self.center_len = 50
        self.Block = Block(self.hidden_dim)

    def forward(self,input_ids,species_ids, w ):
        # input_ids = input_ids.float).long()
        outs = self.pytorch_model(input_ids, species_ids=species_ids )
        x  = outs.embedding # torch.Size([2, 288, 7362]) torch.Size([64, 256, 768]) 11 + 2

        cond = w # shape [B, L, 2]

        # res = self.activation( self.fc2(cond[..., 1].unsqueeze(-1))) * cond[..., 1].unsqueeze(-1)
        # # 通过 fc2 把 w 映射到同维度
        # x = x + res 

        conditions = [ cond[..., 0] ]
        # 用 AdaptiveLayerNorm 做条件调制
        x = self.adaptive_norm(x, conditions=conditions)
        outputs, attn = self.Block(x)

        if self.center_len:
            start_index = (outputs.shape[1] - self.center_len) // 2
            end_index = start_index + self.center_len
            x = outputs[:, start_index: end_index, :]
                
        logits = self.classifier( x.mean(1) ) # torch.Size([4, 50, 13])
        return  logits
    
    def reset_parameters(self):
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 应用到线性模块
        # self.fc2.apply(init_layer)
        self.classifier.apply(init_layer)


lr = 3e-5
batch_size = 64
device_ids = [ 0,1  ] 
device = f"cuda:{device_ids[0]}"
name_list = ["A549", # 1
            "K562",   # 2
            "HepG2", # 3
            "293T", # 4
            ]
    
name = name_list[2] # "A549"

def main():
    epochs = 1 
    patience = 2
    if True:
        dataset = GenomicDataset(   
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/G4_1k.suF.fa",
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/G4_1k.suF_F.tsv",
                
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/G4_1k.suZ.fa",
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/G4_1k.suZ_Z.tsv",
                
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/noG4_1k.suF.fa",
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/noG4_1k.suF_F.tsv",
                
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/noG4_1k.suZ.fa",
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/noG4_1k.suZ_Z.tsv",
                
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/G4_1k.suF_D.tsv",
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/G4_1k.suZ_D.tsv",
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/noG4_1k.suF_D.tsv",
                f"/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/{name}_1k/noG4_1k.suZ_D.tsv", 
            )

        all_f1, all_auc ,all_auprc,  all_mcc = [], [],[],[]
        model = G4former().to(device)
        print(model)
        test_idx = 0
        idx_save_path = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/A549/processed/L1k_1_k_{name}_split_indices.pt'
        if os.path.exists(idx_save_path):
            print(f"Loading existing split indices from {idx_save_path}")
            split_indices = torch.load(idx_save_path)
            train_idx = split_indices["train"]
            val_idx = split_indices["val"]
            test_idx = split_indices["test"]
            print( len(test_idx) )
            
            train_subset = torch.utils.data.Subset(dataset, train_idx)

            val_subset = torch.utils.data.Subset(dataset, val_idx)

            criterion = nn.CrossEntropyLoss(  )

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            model = G4former().to(device)
            model.reset_parameters()
            model = torch.nn.DataParallel(model, device_ids=device_ids)

            optimizer = optim.AdamW(model.parameters(), lr=lr)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

            f1, auc, auprc, mcc, *_ = train_val(model, train_loader, val_loader, criterion,device, optimizer, scheduler, epochs, patience)
            all_f1.append(f1)
            all_auc.append(auc)
            all_auprc.append(auprc)
            all_mcc.append(mcc)

        f1_mean = np.mean(all_f1)
        f1_var = np.var(all_f1)
        auc_mean = np.nanmean(all_auc)
        auc_var = np.nanvar(all_auc)

        # 打印结果
        print(f"\nF1均值: {f1_mean:.4f}, 方差: {f1_var:.4f}")
        print(f"AUC均值: {auc_mean:.4f}, 方差: {auc_var:.4f}")

        print(f"\nAverage F1: {np.mean(all_f1):.4f}, Average AUC: {np.nanmean(all_auc):.4f}, Average AUPRC: {np.nanmean(all_auprc):.4f}, Average MCC: {np.nanmean(all_mcc):.4f}")

        test_subset = torch.utils.data.Subset(dataset, test_idx)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        model = G4former().to(device)
        if name == "293T":
            best_model_wts = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/ck/results/save_L1k_-11-56_293T_epoch_1_64_{lr}_{name}.pth'
        else:
            best_model_wts = f'/home/hjzhang/dataset/home-1/ylxiong/Center/1k/ck/results/save_L1k_-11-56_epoch_1_64_{lr}_{name}.pth'
        if best_model_wts:

            state_dict = torch.load(best_model_wts, map_location=device)            
            new_state_dict = {}
            for key, value in state_dict.items():
                # 移除开头的"module."
                new_key = key[7:] if key.startswith('module.') else key
                new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.eval()
        species_token = 27
        
        val_accs, val_losses = [] ,[]
        # ---------------------- 测试 ----------------------
        loop = tqdm(test_loader, desc=f"[Test]")
        if True:
            val_loss, val_correct, val_total = 0, 0, 0
            val_preds, val_targets, val_probs = [], [], []
            with torch.no_grad():
                for input_ids, w, y in loop:
                    input_ids, w, y = input_ids.to(device), w.to(device), y.to(device)

                    species_ids = torch.tensor([species_token] * len(y), dtype=torch.long).to(device)
                    out = model(input_ids, species_ids,  w)

                    loss = criterion(out, y)
                    val_loss += loss.item() * y.size(0)

                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    pred_labels = out.argmax(1).cpu().numpy()
                    val_preds.extend(pred_labels)
                    val_targets.extend(y.cpu().numpy())
                    val_probs.extend(probs)
                    val_correct += (pred_labels == y.cpu().numpy()).sum()
                    val_total += y.size(0)
            val_losses.append(val_loss / val_total)

            val_accs.append(val_correct / val_total)
            pearson_corr, _ = pearsonr(val_targets, val_probs)
            f1 = f1_score(val_targets, val_preds)
            try:
                auc = roc_auc_score(val_targets, val_probs)
            except:
                auc = float('nan')
            try:
                auprc = average_precision_score(val_targets, val_probs)
            except:
                auprc = float('nan')
            mcc = matthews_corrcoef(val_targets, val_preds)

            print( f"Test Acc={val_accs[-1]:.4f}, ", f"Loss ={val_losses[-1]:.4f}, ",

                f"F1={f1:.4f}, AUC={auc:.4f}, AUPRC={auprc:.4f}, MCC={mcc:.4f}, r={pearson_corr:.4f}")

        
if __name__ == "__main__":
    main()
