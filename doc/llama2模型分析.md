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

不存在bias时 

$`vocab\_size*dim + (12*dim^2+2*dim) * layers + dim+dim*vocab\_size`$

## llama-2-7b
```
{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}
```
存在bias时

$`vocab\_size*4096 + (12*4096^2 +8*4096 + 8*4096/3) * 32 + 4096+(4096*vocab\_size + 4096)=6,442,450,944+4096`$

不存在bias时 

$`vocab\_size*4096 + (12*4096^2+2*4096) * 32 + 4096+4096*vocab\_size`=6,442,450,944+4096+4096$
