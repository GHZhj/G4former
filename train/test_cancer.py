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
    # pad = torch.zeros(signals.size(0), 1)  # [N, 1]

    # signals = torch.cat([pad, signals, pad], dim=-1)
    return  signals  

class GenomicDataset(Dataset):
    def __init__(self, fasta_pos_fu, methylation_pos_fusame, 
    fasta_neg_fu, methylation_neg_fusame, 
    Dnase_Z_F,Dnase_F_F ,name
   
    ):
        # methylation_pos_fudif = _read_signal_file(methylation_pos_fusame )
        # # methylation_pos_zdif = _read_signal_file(methylation_pos_zsame)
        # methylation_neg_fudif = _read_signal_file(methylation_neg_fusame)
        # # methylation_neg_zdif = _read_signal_file(methylation_neg_zsame)

        # self.methylation = np.concatenate([
        #                     methylation_pos_fudif,   
        #                     torch.flip(methylation_neg_fudif, dims=[1]) , 
        #                     ], axis=0)
        pos_seqs_fu, pos_chr_fu = self.read_fasta(fasta_pos_fu)
        # pos_seqs_z, pos_chr_z = self.read_fasta(fasta_pos_z)
        neg_seqs_fu, neg_chr_fu = self.read_fasta(fasta_neg_fu)
        # neg_seqs_z, neg_chr_z = self.read_fasta(fasta_neg_z)
        # self.fasta_pos = pos_seqs_fu + pos_seqs_z
        # self.fasta_neg = neg_seqs_fu + neg_seqs_z
        self.chr_pos = pos_chr_fu + neg_chr_fu
        # self.chr_neg = neg_chr_fu + neg_chr_z
        sequences = pos_seqs_fu + neg_seqs_fu
 
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
        # idx_save_path = f'/home/hjzhang/cancer/processed/L1k_1_k_{name}_split_indices.pt'
        # if not os.path.exists(idx_save_path):
        #     split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
        #     torch.save(split_indices, idx_save_path)
        #     print(f"Saved split indices to {idx_save_path}")

        # 初始化 tokenizer
        tokenizer = get_ntv3_tokenizer()

        # tokenizer 输出 numpy array
        tokens_np = tokenizer.batch_np_tokenize(sequences)  # shape: [batch, seq_len]
        self.input_ids = torch.tensor(tokens_np, dtype=torch.long)

        epi_path = f'/home/hjzhang/cancer/processed/L1k_1_k_{name}_epis_WGBS_3_S.pt'
        if os.path.exists(epi_path):
            print(f"{epi_path} already exists. Loading compressed features...")
            self.epi = torch.load(epi_path).float()#  * mask.unsqueeze(-1) 
            print(f"Saved compressed epi features to no7")

            print(self.epi.max())
        else:
            print("Computing and saving compressed epi features...")
            epi_tensor = torch.tensor( self.methylation[:, :, np.newaxis]  , dtype=torch.float32)
            torch.save(epi_tensor, epi_path)
            self.epi = epi_tensor.float() 
            print(f"Saved compressed epi features to {epi_path}")
            print(self.epi.max())

            
        # Dnase_Z_F = _read_signal_file(Dnase_Z_F)
        # # Dnase_Z_Z = _read_signal_file(Dnase_Z_Z)
        # Dnase_F_F = _read_signal_file(Dnase_F_F)
        # # Dnase_F_Z = _read_signal_file(Dnase_F_Z)
        # self.Dnase = np.concatenate([
        #                     Dnase_Z_F,  
        #                     torch.flip(Dnase_F_F, dims=[1]) , 
        #                     ], axis=0)
        Dnase_path = f'/home/hjzhang/cancer/processed/L1k_1_k_{name}_epis_atac_S2.pt'
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
            print(self.Dnase.max())


    def read_fasta(self, file):
        seqs = []
        chroms = []
        with open(file) as f:
            seq = ''
            chr_id = None
            for line in f:
                if line.startswith('>'):
                    # Parse chromosome ID between '>' and ':'
                    if seq and chr_id is not None:
                        seqs.append(seq.upper())
                        chroms.append(chr_id)
                        seq = ''
                    header = line.strip()
                    if ':' in header:
                        chr_id = header[1:].split(':')[0]
                    else:
                        chr_id = header[1:]
                else:
                    seq += line.strip()
            if seq and chr_id is not None:
                seqs.append(seq.upper())
                chroms.append(chr_id)
        return seqs, chroms


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        train_seq = torch.cat([ self.Dnase[idx],  self.epi[idx]],-1) # 添加首尾

        return (
            self.input_ids[idx] , train_seq ,
            self.chr_pos[idx])
        

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
        self.fc2 = nn.Linear(1, 768)  # Second fully connected layer (optional)
        self.hidden_dim = 768

        self.classifier = nn.Linear(self.hidden_dim, 2)  # Binary classification
        self.activation = nn.GELU()  # Activation function

        self.pytorch_model = AutoModel.from_pretrained(
                "/home/hjzhang/ntv3G4/nucleotide_transformer_v3/NTv3_100m_post",
                    trust_remote_code=True
                )

        self.center_len = 50
        self.Block = Block(self.hidden_dim)

    def forward(self,input_ids,species_ids,  w ):
        outs = self.pytorch_model(input_ids, species_ids=species_ids )
        x  = outs.embedding # torch.Size([2, 288, 7362]) torch.Size([64, 256, 768]) 11 + 2

        cond = w # shape [B, L, 2]
        
        mask = ( input_ids == 7).float() #
        wgbs = cond[..., 1].unsqueeze(-1) * mask.unsqueeze(-1) 

        res = self.activation( self.fc2( wgbs )) * wgbs #  .unsqueeze(-1)
        # 通过 fc2 把 w 映射到同维度
        x = x + res 

        conditions = [ cond[..., 0]]
        # 用 AdaptiveLayerNorm 做条件调制
        x = self.adaptive_norm(x, conditions=conditions)
        outputs, attn = self.Block(x)
        if self.center_len:
            start_index = (outputs.shape[1] - self.center_len) // 2
            end_index = start_index + self.center_len
            x = outputs[:, start_index: end_index, :]
            start_index3 = (outputs.shape[1] - 50) // 2
            end_index3 = start_index3 + 50

        attn = attn[:, :, start_index3: end_index3,start_index3: end_index3]
        
        logits = self.classifier(x.mean(1))
        return logits, attn
    
    def reset_parameters(self):
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 递归应用到所有子模块
        self.fc2.apply(init_layer)
        self.classifier.apply(init_layer)
 
lr = 3e-5
batch_size = 4*72
device_ids = [  0,1 ] 
device = f"cuda:{device_ids[0]}"

name_list = [
    "ACC_tumor", "BLCA_tumor","BRCA_tumor","COAD_tumor","ESCA_tumor","HNSC_tumor","KIRC_tumor","KIRP_tumor","LGG_tumor","LIHC_tumor",
    "LUAD_tumor","LUSC_tumor","MESO_tumor","PCPG_tumor","PRAD_tumor","SKCM_tumor","STAD_tumor","TGCT_tumor","THCA_tumor","UCEC_tumor",


    "ACC_normal", "BLCA_normal","BRCA_normal","COAD_normal","ESCA_normal","HNSC_normal","KIRC_normal","KIRP_normal","LGG_normal","LIHC_normal",
    "LUAD_normal","LUSC_normal","MESO_normal","PCPG_normal","PRAD_normal","SKCM_normal","STAD_normal","TGCT_normal","THCA_normal","UCEC_normal",
]


best_model_wts = f'/home/hjzhang/ntv3G4/1save_L1k_atacpeak-epoch_1_64_3e-05_HepG2.pth'

        
def main():
    epochs = 1 

    if best_model_wts:
        state_dict = torch.load(best_model_wts, map_location=device)            
        new_state_dict = {}
        for key, value in state_dict.items():
                        # 移除开头的"module."
            new_key = key[7:] if key.startswith('module.') else key
            new_state_dict[new_key] = value
        model = G4former().to(device)

        model.load_state_dict(new_state_dict)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.eval()

        species_token = 27
        for name in name_list:
            dataset = GenomicDataset(   
                f"/home/hjzhang/{name}/dataZ.su.fa",
                f"/home/hjzhang/{name}/dataZ.su_W.tsv", # WGBS

                f"/home/hjzhang/{name}/dataF.su.fa",
                f"/home/hjzhang/{name}/dataF.su_W.tsv", # WGBS
                
                f"/home/hjzhang/{name}/dataZ.su_A.tsv", # ATAC-seq (100-bp)
                f"/home/hjzhang/{name}/dataF.su_A.tsv", # ATAC-seq (100-bp)
                name
                )

            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


            all_probs, all_chr = [] , []
            loop = tqdm(test_loader, desc=f"[Test]")
            with torch.no_grad():
                for input_ids, w, chrname in loop:
                    input_ids, w = input_ids.to(device),  w.to(device)

                    species_ids = torch.tensor([species_token] * len(w), dtype=torch.long).to(device)
                    out, _ = model(input_ids, species_ids, w)

                    probs = torch.softmax(out, dim=1)[:, 1]  # 正类概率
                    all_probs.append(probs.cpu())
                    all_chr.extend(chrname)   # ⭐ 保存chrname

            # 拼接所有 batch
            all_probs = torch.cat(all_probs).numpy()

            # 保存
            import numpy as np

            all_chr = np.array(all_chr)

            np.savez(f"_{name}_val_probs_with_chr.npz",
            probs=all_probs,
            chr=all_chr)

        
if __name__ == "__main__":
    main()
