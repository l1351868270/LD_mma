# 代码
Adapted from https://github.com/facebookresearch/llama/blob/main/llama/model.py

# 模型参数量
## self-attention
### attention_norm
只有一个参数self.weight，参数量为 

$dim$
### attention
attention：有四个参数self.wq, self.wk, self.wv, self.wo，每个权重的参数相同

存在bias时  

$`4*dim^2+4*dim`$  

不存在bias时 

$`4*dim^2`$ 

### self-attention total
存在bias时 

$`4*dim^2+5*dim`$ 

不存在bias时 

$`4*dim^2+dim`$
## mlp
### ffn_norm
只有一个参数self.weight，参数量为

 $dim$
### ffn
有三个参数self.w1, self.w2, self.w3，参数 $`hiddenDim=2*4*dim/3=8*dim/3`$ 

存在bias时 

$`dim * hiddenDim + dim + hiddenDim * dim + hiddenDim + dim * hiddenDim + dim`$

$` = 3*hiddenDim*dim + 2*dim + hiddenDim`$

$` = 8*dim/3 * dim *3+ 2*dim + 8*dim/3 = 8*dim^2 + 2*dim + 8*dim/3`$

不存在bias时 

$` 8*dim/3 * dim * 3 = 8*dim^2`$

### ffn total
存在bias时 

$`8*dim/3 * dim * 3 + 3*dim + 8*dim/3 = 8*dim^2 + 3*dim + 8*dim/3`$

不存在bias时 

$` 8*dim/3 * dim * 3 + dim = 8*dim^2 + dim `$

## transformer/per
存在bias时 

$`4*dim^2+5*dim + 8*dim/3 * dim * 3 + 3*dim + 8*dim/3`$

$`= 4*dim^2+8*dim/3*dim*3+8*dim + 8*dim/3`$ 

$`= 12*dim^2 +8*dim + 8*dim/3`$

不存在bias时 

$` 4*dim^2+dim + 8*dim/3 * dim * 3 + dim = 12*dim^2+2*dim `$

## llama-2
### self.tok_embeddings

$`vocab\_size*dim`$

### self.layers
存在bias时

$`(12*dim^2 +8*dim + 8*dim/3) * layers`$

不存在bias时 

$`(12*dim^2+2*dim) * layers`$

### self.norm

$dim$

### self.output
存在bias时

$`dim*vocab\_size + dim`$

不存在bias时 

$`dim*vocab\_size`$

### llama-2 total

存在bias时

$`vocab\_size*dim + (12*dim^2 +8*dim + 8*dim/3) * layers + dim+(dim*vocab\_size + dim)`$

$`=2*vocab\_size*dim + (12*dim^2 +8*dim + 8*dim/3) * layers + 2*dim`$

$`=12*layers*dim^2 + (2*vocab\_size+8*layers+8/3*layers+2)*dim`$

$`\approx 12*layers*dim^2`$

不存在bias时 

$`vocab\_size*dim + (12*dim^2+2*dim) * layers + dim+dim*vocab\_size`$

$`=12*layers*dim^2+(2*vocab\_size+2*layers+1)*dim`$

$`\approx 12*layers*dim^2`$

## llama-2-7b
```
{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}
```
存在bias时

$`32000*4096 + (12*4096^2 +8*4096 + 8*4096/3) * 32 + 4096+(4096*32000 + 4096)`$

$`=131,072,000 + 6,442,450,944+4096 + 4096 + (131,072,000 + 4096)`$

$`=6,704,607,232`$

$`\approx 6,442,450,944`$

不存在bias时 

$`32000*4096 + (12*4096^2+2*4096) * 32 + 4096+4096*32000`$

$`=131,072,000+6,442,450,944+4096+131,072,000`$
$`=6,704,599,040`$
$`\approx 6,442,450,944`$

# FLOPs估计
## matmul FLOPs
### self-attention matmul
假设输入bsz, seqlen
#### attention_norm matmul
没有矩阵乘法
#### attention matmul
xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)：

$`bsz * seqlen * dim * dim * 2 * 3 = 6 * bsz * seqlen * dim^2`$


torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim):

$` bsz * seqlen * dim * seqlen * 2 = 2 * bsz * seqlen^2 * dim`$

torch.matmul(scores, values):

$`bsz * seqlen * seqlen * dim * 2 = 2 * bsz * seqlen^2 * dim`$

self.wo(output):

$`bsz * seqlen * dim * dim * 2 = 2 * bsz * seqlen * dim^2`$

#### self-attention total matmul
$`bsz * seqlen * dim * dim * 2 * 3 + bsz * seqlen * dim * seqlen * 2 + bsz * seqlen * seqlen * dim * 2 +  2 * bsz * seqlen * dim^2`$

$`= 8 * bsz * seqlen * dim^2 + 4 * bsz * seqlen^2 * dim`$

### mlp matmul
#### ffn_norm matmul
没有矩阵乘法
#### ffn matmul
self.w1(x): 

$` bsz * seqlen * dim * hidden_dim * 2`$

self.w3(x):

$` bsz * seqlen * dim * hidden_dim * 2`$

self.w2():

$` bsz * seqlen * hidden_dim * dim  * 2`$

#### ffn total matmul

$` 6 * bsz * seqlen * hidden_dim * dim`$ 

$` 16 * bsz * seqlen * dim ^ 2`$ 

### transformer/per matmul
$`8 * bsz * seqlen * dim^2 + 4 * bsz * seqlen^2 * dim + 16 * bsz * seqlen * dim ^ 2`$

$`=24 * bsz * seqlen * dim^2 + 4 * bsz * seqlen^2 * dim`$

### llama-2 matmul
#### self.tok_embeddings matmul
无矩阵乘法
#### self.layers matmul

$`(24 * bsz * seqlen * dim^2 + 4 * bsz * seqlen^2 * dim) * layers`$

#### self.norm matmul
无矩阵乘法 
#### self.output matmul

$` bsz * seqlen * dim * vocab\_size * 2`$

$` = 2 * bsz * seqlen * dim * vocab\_size`$

#### llama-2 total matmul
$`(24 * bsz * seqlen * dim^2 + 4 * bsz * seqlen^2 * dim) * layers + 2 * bsz * seqlen * dim * vocab\_size `$

## llama-2-7b
```
{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 32000}
```

$`(24 * bsz * seqlen * 4096^2 + 4 * bsz * seqlen^2 * 4096) * 32 + 2 * bsz * seqlen * 4096 * 32000 `$

$`= bsz * seqlen *  ((24 * 4096^2 + 4 * seqlen * 4096) * 32 + 2 * 4096 * 32000) `$

假设bsz=1， seqlen=1

$`bsz * seqlen *  ((24 * 4096^2 + 4 * seqlen * 4096) * 32 + 2 * 4096 * 32000) `$


$`(24 * 4096^2 + 4 * 4096) * 32 + 2 * 4096 * 32000 `$ 

$` = 13,147,570,176 `$ 

假设bsz=1， seqlen=256

$`256 *  ((24 * 4096^2 + 4 * 256 * 4096) * 32 + 2 * 4096 * 32000)`$

$` = 3,400,003,485,696 = 3.4TFLOAPs`$ 

假设bsz=1， seqlen=1024

$`1024 *  ((24 * 4096^2 + 4 * 1024 * 4096) * 32 + 2 * 4096 * 32000)`$

$` = 14,012,330,803,200 = 14TFLOAPs`$ 

312TFLOAPs

## 非matmul FLOPs