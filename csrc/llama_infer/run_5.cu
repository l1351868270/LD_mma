/*
Adapted from https://github.com/karpathy/llama2.c/blob/master/run.c

ncu -k rmsnorm -f --set full -o run_5 ./run_5 stories42M_fp16.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"

ncu -k rmsnormV2 --csv --log-file run_5_rmsnorm.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./run_5 stories15M_fp16.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
python stat-csv.py run_5_rmsnorm.csv --kernels "rmsnormV2"
['rmsnormV2']
kernel, mean(us), std, med, num
rmsnormV2,  4.932,  0.179,  4.864,  325

ncu -k ropeV2 --csv --log-file run_5_ropeV2.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./run_5 stories15M_fp16.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
python stat-csv.py run_5_ropeV2.csv --kernels "ropeV2"
['ropeV2']
kernel, mean(us), std, med, num
ropeV2,  3.373,  0.181,  3.424,  1936

ncu -k residual_connectionV2 --csv --log-file run_5_residual_connectionV2.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./run_5 stories15M_fp16.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
python stat-csv.py run_5_residual_connectionV2.csv --kernels "residual_connectionV2"
['residual_connectionV2']
kernel, mean(us), std, med, num
residual_connectionV2,  2.217,  0.131,  2.208,  3936

ncu -k swigluV2 --csv --log-file run_5_swigluV2.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./run_5 stories15M_fp16.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
python stat-csv.py run_5_swigluV2.csv --kernels "swigluV2"
['swigluV2']
kernel, mean(us), std, med, num
swigluV2,  2.662,  0.110,  2.656,  1856

ncu -k multihead_attentionV2 --csv --log-file run_5_multihead_attentionV2.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./run_5 stories15M_fp16.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
python stat-csv.py run_5_multihead_attentionV2.csv --kernels "multihead_attentionV2"

nsys profile --stats=true ./run_5 llama2_7b_fp16.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
Inference for Llama-2 Transformer model in pure C 
nvcc -O3 -arch=sm_86 -o run_5 run_5.cu
./run_5 stories15M_fp16.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
./run_5  llama2_7b_fp16.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <stdint.h>
#include <cublas_v2.h>
#include <vector>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    half* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    half* rms_att_weight; // (layer, dim) rmsnorm weights
    half* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    half* wq; // (layer, dim, n_heads * head_size)
    half* wk; // (layer, dim, n_kv_heads * head_size)
    half* wv; // (layer, dim, n_kv_heads * head_size)
    half* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    half* w1; // (layer, hidden_dim, dim)
    half* w2; // (layer, dim, hidden_dim)
    half* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    half* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    half* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    half *x; // activation at current time stamp (dim,)
    half *xb; // same, but inside a residual branch (dim,)
    half *xb2; // an additional buffer just for convenience (dim,)
    half *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    half *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    half *q; // query (dim,)
    half *k; // key (dim,)
    half *v; // value (dim,)
    half *att; // buffer for scores/attention values (n_heads, seq_len)
    half *logits; // output logits
    // kv cache
    half* key_cache;   // (layer, seq_len, dim)
    half* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    half* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    cudaMalloc((void**)&s->x, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb2, p->dim * sizeof(half));
    cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->q, p->dim * sizeof(half));
    cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(half));
    cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(half));
    cudaMalloc((void**)&s->att, p->n_heads * p->seq_len * sizeof(half));
    cudaMalloc((void**)&s->logits, p->vocab_size * sizeof(half));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->xb2);
    cudaFree(s->hb);
    cudaFree(s->hb2);
    cudaFree(s->q);
    cudaFree(s->att);
    cudaFree(s->logits);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, half* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, half** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    printf("dim: %d\n", config->dim);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (half*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    half* weights_ptr = *data + sizeof(Config)/sizeof(half);
    // for (int i = 0; i < 2 * config->dim; i++) {
    //     printf("weights_ptr[%d] = %f\n", i, __half2float(*(weights_ptr+i)));
    // }
    // memory_map_weights(weights, config, weights_ptr, shared_weights);
    void *device_memory;
    cudaMalloc((void**)&device_memory, *file_size);
    cudaMemcpy(device_memory, weights_ptr, *file_size, cudaMemcpyHostToDevice);
    memory_map_weights(weights, config, (half*)device_memory, shared_weights);
    
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

__global__
void rmsnorm(half* o, half* x, half* weight, int size) {
    // calculate sum of squares
    half ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    // __shfl_xor_sync(uint32_t(-1), x[0], 1);
    // printf("ss = %f\n", __half2float(ss));
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // printf("ss = %f\n", __half2float(ss));
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
        // printf("o[%d] = %f\n", j, __half2float(o[j]));
    }
}

__global__
void rmsnormV2(half* o, half* x, half* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    int l = size / blockDim.x;
    #pragma unroll
    for (int i = 0; i < l; i++) {
        ss += __half2float(x[threadIdx.x * l + i]) * __half2float(x[threadIdx.x * l + i]);
    }

    // printf("idx=%d, ss=%f\n", threadIdx.x, ss);
    #pragma unroll
    for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
        ss += __shfl_xor_sync(uint32_t(-1), ss, mask);
    }
    // printf("threadx=%d, ss = %f\n", threadIdx.x, __half2float(ss));
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    
    #pragma unroll
    for (int j = 0; j < l; j++) {
        o[threadIdx.x * l + j] = weight[threadIdx.x * l + j] * (__float2half(ss) * x[threadIdx.x * l + j]);
    }
}

__global__
void rmsnormV3(half* o, half* token_embedding_table, int *token, half* weight, int size) {
    // calculate sum of squares
    half *x = token_embedding_table + (*token) * size;
    float ss = 0.0f;
    int l = size / blockDim.x;
    #pragma unroll
    for (int i = 0; i < l; i++) {
        ss += __half2float(x[threadIdx.x * l + i]) * __half2float(x[threadIdx.x * l + i]);
    }

    // printf("idx=%d, ss=%f\n", threadIdx.x, ss);
    #pragma unroll
    for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
        ss += __shfl_xor_sync(uint32_t(-1), ss, mask);
    }
    // printf("threadx=%d, ss = %f\n", threadIdx.x, __half2float(ss));
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    
    #pragma unroll
    for (int j = 0; j < l; j++) {
        o[threadIdx.x * l + j] = weight[threadIdx.x * l + j] * (__float2half(ss) * x[threadIdx.x * l + j]);
    }
}

__global__
void softmax(half* x, int size) {
    // find max value (for numerical stability)
    half max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    half sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

__device__
void device_softmax(half* x, int size) {
    // find max value (for numerical stability)
    half max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    half sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

__global__
void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

__global__
void matmul(half* xout, half* x, half* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    // #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        half val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void matmulV2(cublasHandle_t* handle, half* xout, half* x, half* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int M = 1;
    int N = d;
    int K = n;

    half alpha = half(1.f);
    half beta = half(0.f);
    cublasStatus_t ret = cublasHgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N,
          	  N, M, K,
          	  &alpha,
          	  w, K,
          	  x, K,
          	  &beta,
          	  xout, N);
    
}

__global__
void rope(half* q, half* k, int dim, int kv_dim, int head_size, int pos) {
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            half freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            half val = (half)pos * freq;
            half fcr = cosf(val);
            half fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                half* vec = v == 0 ? q : k; // the vector to rotate (query or key)
                half v0 = vec[i];
                half v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
                // printf("vel_i: %f\n", (float)vel_i);
            }
        }
}

__global__
void ropeV2(half* q, half* k, int dim, int kv_dim, int head_size, int pos) {
    int i = 2 * (threadIdx.x + blockDim.x * blockIdx.x);
        // for (int i = 0; i < dim; i+=2) {
    int head_dim = i % head_size;
    half freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    half val = (half)pos * freq;
    half fcr = cosf(val);
    half fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
    for (int v = 0; v < rotn; v++) {
        half* vec = v == 0 ? q : k; // the vector to rotate (query or key)
        half v0 = vec[i];
        half v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i+1] = v0 * fci + v1 * fcr;
                // printf("vel_i: %f\n", (float)vel_i);
    }
}

__global__ 
void multihead_attention(half* s_att, half* s_q, half* s_key_cache, half* s_value_cache, half* s_xb,
                         int n_heads, int head_size, int seq_len, int pos, int loff, int kv_dim, int kv_mul) {
        for (int h = 0; h < n_heads; h++) {
            // get the query vector for this head
            half* q = s_q + h * head_size;
            // attention scores for this head
            half* att = s_att + h * seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                half* k = s_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                half score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            device_softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            half* xb = s_xb + h * head_size;
            memset(xb, 0, head_size * sizeof(half));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                half* v = s_value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                half a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
}

// __global__ 
// void multihead_attentionV2(half* s_att, half* s_q, half* s_key_cache, half* s_value_cache, half* s_xb,
//                          int n_heads, int head_size, int seq_len, int pos, int loff, int kv_dim, int kv_mul) {
//     int h = blockIdx.x;
//     // get the query vector for this head
//     half* q = s_q + h * head_size;
//     // attention scores for this head
//     half* att = s_att + h * seq_len;
//     // iterate over all timesteps, including the current one

//     // int t = threadIdx.x;

//     // int l = pos / blockDim.x + 1;
//     // if (pos > t) {

//     // }

//     for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
//         // get the key vector for this head and at this timestep
//         half* k = s_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
//         // calculate the attention score as the dot product of q and k
//         half score = 0.0f;
//         for (int i = 0; i < head_size; i++) {
//             score += q[i] * k[i];
//         }
//         score /= sqrtf(head_size);
//         // save the score to the attention buffer
//         att[t] = score;
//     }
//     __syncthreads();
//     // softmax the scores to get attention weights, from 0..pos inclusively

//     if (threadIdx.x == 0) {
//         device_softmax(att, pos + 1);
//     }

//     // printf("blockIdx: %d\n", blockIdx.x);
    
//     // weighted sum of the values, store back into xb
//     half* xb = s_xb + h * head_size;
//     memset(xb, 0, head_size * sizeof(half));
//     __syncthreads();
//     // if (threadIdx.x == 0) {
//     for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
//         for (int t = 0; t <= pos; t += 1) {
//             // get the value vector for this head and at this timestep
//             half* v = s_value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
//             // get the attention weight for this timestep
//             half a = att[t];
//             // accumulate the weighted value into xb
        
//             xb[i] += a * v[i];
//         }
//     }
//     // }
// }

__global__ 
void multihead_attentionV2(half* s_att, half* s_q, half* s_key_cache, half* s_value_cache, half* s_xb,
                         int n_heads, int head_size, int seq_len, int pos, int loff, int kv_dim, int kv_mul) {
    int h = blockIdx.x;
    // get the query vector for this head
    half* q = s_q + h * head_size;
    // attention scores for this head
    half* att = s_att + h * seq_len;
    // iterate over all timesteps, including the current one


    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        half* k = s_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        half score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
    }
    // softmax the scores to get attention weights, from 0..pos inclusively


    device_softmax(att, pos + 1);


    // printf("blockIdx: %d\n", blockIdx.x);
    
    // weighted sum of the values, store back into xb
    half* xb = s_xb + h * head_size;
    memset(xb, 0, head_size * sizeof(half));

            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                half* v = s_value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                half a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
}

__global__
void residual_connection(half* x, half* src, int dim) {
        for (int i = 0; i < dim; i++) {
            x[i] += src[i];
        }
}

__global__
void residual_connectionV2(half* x, half* src, int dim) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    x[i] += src[i];
}

__global__ 
void swiglu(half* s_hb, half* s_hb2, int hidden_dim) {
        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            half val = s_hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s_hb2[i];
            s_hb[i] = val;
        }
}

__global__ 
void swigluV2(half* s_hb, half* s_hb2, int hidden_dim) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    half val = s_hb[i];
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val *= (1.0f / (1.0f + expf(-val)));
    // elementwise multiply with w3(x)
    val *= s_hb2[i];
    s_hb[i] = val;
}

__global__ 
void print_vector(half* x, int dim) {
        for (int i = 0; i < dim; i++) {
            printf("%d=%f ", i, __half2float(x[i]));
        }
        printf("\n");
}

__global__ 
void print_vector_int(int* x, int dim) {
        for (int i = 0; i < dim; i++) {
            printf("%d=%d ", i, x[i]);
        }
        printf("\n");
}

__global__ void get_content_row(half *x, half* token_embedding_table, int *token, int dim) {
// __global__ void get_content_row() {
    for (int i = 0; i<dim; i++) {
        x[i] = *(token_embedding_table + (*token) * dim + i);
    }

    // printf("get_content_row\n");
}

__global__ void static_kernel(half *x, half* token_embedding_table, int *token, int dim) {
// __global__ void get_content_row() {
    // for (int i = 0; i<dim; i++) {
    //     x[i] = *(token_embedding_tableint + (*token) * dim + i);
    // }
    printf("static_kernel x=%f, token_embedding_table=%f, token=%d, dim=%d\n", __half2float(x[0]), __half2float(token_embedding_table[0]), *token, dim);
}

__global__ void clockBlock(clock_t clock_count) {
printf("clockBlock clock_count=%d\n", clock_count);
}

// void create_global_graph(cudaGraphExec_t *graph_exec, cudaGraph_t *graph, cudaStream_t *stream, cudaGraphNode_t *blockDeviceNode) {
//     cudaError_t err;
//     cudaGraphNode_t allocNodeA, freeNodeA;
// //   cudaGraphNode_t
//     std::vector<cudaGraphNode_t> graph_nodes;
//     cudaMemAllocNodeParams allocParams;
//     cudaKernelNodeParams blockDeviceNodeParams = {0};

//     // cudaGraphExec_t graph_exec;
//     // cudaGraph_t graph;
//     // cudaGraphCreate(&graph, 0);
//     // cudaStream_t stream;
//     // cudaStreamCreate(&stream);

//     float kernelTime = 5;  // time for each thread to run in microseconds

//     int dim = 512;

//     void *blockDeviceArgs[1] = {(void *)&dim};

//     blockDeviceNodeParams.gridDim = dim3(1, 1, 1);
//     blockDeviceNodeParams.blockDim = dim3(1, 1, 1);
//     blockDeviceNodeParams.sharedMemBytes = 0;
//     blockDeviceNodeParams.extra = NULL;
//     blockDeviceNodeParams.func = (void *)static_kernel;
//     blockDeviceNodeParams.kernelParams = (void **)blockDeviceArgs;

//     err = cudaGraphAddKernelNode(blockDeviceNode, *graph, NULL,
//                                          0, &blockDeviceNodeParams);

//     graph_nodes.push_back(*blockDeviceNode);
//     if (err != cudaSuccess) {
//         printf("cudaGraphAddKernelNode with error: %s\n", cudaGetErrorString(err));
//     }

//     err = cudaGraphInstantiate(graph_exec, *graph, nullptr, nullptr, 0);
//     if (err != cudaSuccess) {
//         printf("cudaGraphInstantiate with error: %s\n", cudaGetErrorString(err));
//     }

//     int tmp = 256;
//     void *kernelArgs[1] = {(void *)&tmp};

//     blockDeviceNodeParams.kernelParams = kernelArgs;

//     err = cudaGraphExecKernelNodeSetParams(*graph_exec, *blockDeviceNode, &blockDeviceNodeParams);
//     if (err != cudaSuccess) {
//             printf("cudaGraphExecKernelNodeSetParams with error: %s\n", cudaGetErrorString(err));
//     }


//     err = cudaGraphLaunch(*graph_exec, *stream);
//     if (err != cudaSuccess) {
//         printf("cudaGraphLaunch with error: %s\n", cudaGetErrorString(err));
//     }
//     err = cudaStreamSynchronize(*stream);
//     if (err != cudaSuccess) {
//         printf("cudaStreamSynchronize with error: %s\n", cudaGetErrorString(err));
//     }

// //   return graph_nodes;
// }

half* forward(cublasHandle_t* handle, cudaGraph_t *graph, cudaGraphExec_t *graph_exec, cudaStream_t* stream, bool* graphCreated, cudaGraphNode_t *blockDeviceNode,
              Transformer* transformer, int *token, int htoken, int pos) {

    
    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    half *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    cudaError_t err;

    cudaGraphNode_t blockDeviceNode_1;
    if (!(*graphCreated)) {
        *graphCreated= true;
        cudaKernelNodeParams blockDeviceNodeParams = {0};

        void *blockDeviceArgs[4] = {(void *)&x, (void *)(&(w->token_embedding_table)), (void *)&token, (void *)&dim};

        blockDeviceNodeParams.gridDim = dim3(1, 1, 1);
        blockDeviceNodeParams.blockDim = dim3(1, 1, 1);
        blockDeviceNodeParams.sharedMemBytes = 0;
        blockDeviceNodeParams.extra = NULL;
        blockDeviceNodeParams.func = (void *)get_content_row;
        blockDeviceNodeParams.kernelParams = (void **)blockDeviceArgs;

        err = cudaGraphAddKernelNode(blockDeviceNode, *graph, NULL,
                                         0, &blockDeviceNodeParams);

        if (err != cudaSuccess) {
            printf("cudaGraphAddKernelNode with error: %s\n", cudaGetErrorString(err));
        }

        cudaKernelNodeParams blockDeviceNodeParams_1 = {0};

        void *blockDeviceArgs_1[4] = {(void *)&x, (void *)(&(w->token_embedding_table)), (void *)&token, (void *)&dim};

        blockDeviceNodeParams_1.gridDim = dim3(1, 1, 1);
        blockDeviceNodeParams_1.blockDim = dim3(1, 1, 1);
        blockDeviceNodeParams_1.sharedMemBytes = 0;
        blockDeviceNodeParams_1.extra = NULL;
        blockDeviceNodeParams_1.func = (void *)static_kernel;
        blockDeviceNodeParams_1.kernelParams = (void **)blockDeviceArgs_1;

        err = cudaGraphAddKernelNode(blockDeviceNode + 1, *graph, blockDeviceNode,
                                         1, &blockDeviceNodeParams_1);

        if (err != cudaSuccess) {
            printf("cudaGraphAddKernelNode with error: %s\n", cudaGetErrorString(err));
        }

        err = cudaGraphInstantiate(graph_exec, *graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            printf("cudaGraphInstantiate with error: %s\n", cudaGetErrorString(err));
        }
    } else {
        cudaKernelNodeParams blockDeviceNodeParams = {0};

        void *blockDeviceArgs[4] = {(void *)&x, (void *)(&(w->token_embedding_table)), (void *)&token, (void *)&dim};

        blockDeviceNodeParams.gridDim = dim3(1, 1, 1);
        blockDeviceNodeParams.blockDim = dim3(1, 1, 1);
        blockDeviceNodeParams.sharedMemBytes = 0;
        blockDeviceNodeParams.extra = NULL;
        blockDeviceNodeParams.func = (void *)get_content_row;
        blockDeviceNodeParams.kernelParams = (void **)blockDeviceArgs;
  
        err = cudaGraphExecKernelNodeSetParams(*graph_exec, *blockDeviceNode, &blockDeviceNodeParams);
        if (err != cudaSuccess) {
            printf("cudaGraphExecKernelNodeSetParams with error: %s\n", cudaGetErrorString(err));
        }

        cudaKernelNodeParams blockDeviceNodeParams_1 = {0};

        void *blockDeviceArgs_1[4] = {(void *)&x, (void *)(&(w->token_embedding_table)), (void *)&token, (void *)&dim};

        blockDeviceNodeParams_1.gridDim = dim3(1, 1, 1);
        blockDeviceNodeParams_1.blockDim = dim3(1, 1, 1);
        blockDeviceNodeParams_1.sharedMemBytes = 0;
        blockDeviceNodeParams_1.extra = NULL;
        blockDeviceNodeParams_1.func = (void *)static_kernel;
        blockDeviceNodeParams_1.kernelParams = (void **)blockDeviceArgs_1;
  
        err = cudaGraphExecKernelNodeSetParams(*graph_exec, *(blockDeviceNode+1), &blockDeviceNodeParams_1);
        if (err != cudaSuccess) {
            printf("cudaGraphExecKernelNodeSetParams with error: %s\n", cudaGetErrorString(err));
        }
    }

    err = cudaGraphLaunch(*graph_exec, *stream);
    if (err != cudaSuccess) {
        printf("cudaGraphLaunch with error: %s\n", cudaGetErrorString(err));
    }
    err = cudaStreamSynchronize(*stream);
    if (err != cudaSuccess) {
        printf("cudaStreamSynchronize with error: %s\n", cudaGetErrorString(err));
    }
        
    // forward all the layers
    #pragma unroll
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        // rmsnormV2<<<1,32,0,*stream>>>(s->xb, x, w->rms_att_weight + l*dim, dim);

        // cudaStreamBeginCapture(*stream, cudaStreamCaptureModeGlobal);
        rmsnormV2<<<1,32,0,*stream>>>(s->xb, x, w->rms_att_weight + l*dim, dim);

        // printf("rmsnormV2: \n");
        // print_vector<<<1,1>>>(s->xb, dim);
        // cudaDeviceSynchronize();
        // exit(1);
        // cudaStreamSynchronize(*stream);
        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position

        matmulV2(handle, s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        // printf("matmulV2: \n");
        // print_vector<<<1,1>>>(s->q, dim);
        // cudaDeviceSynchronize();
        // exit(1);

        matmulV2(handle, s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmulV2(handle, s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        ropeV2<<<dim/64, 32,0,*stream>>>(s->q, s->k, dim, kv_dim, head_size, pos);
        // printf("ropeV2 q: \n");
        // print_vector<<<1,1>>>(s->q, dim);
        // cudaDeviceSynchronize();
        // printf("ropeV2 k: \n");
        // print_vector<<<1,1>>>(s->k, dim);
        // cudaDeviceSynchronize();
        // exit(1);

        multihead_attentionV2<<<head_size, 1,0,*stream>>>(s->att, s->q, s->key_cache, s->value_cache, s->xb,
                         p->n_heads, head_size, p->seq_len, pos, loff, kv_dim, kv_mul);
        // final matmul to get the output of the attention
        matmulV2(handle, s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        // for (int i = 0; i < dim; i++) {
        //     x[i] += s->xb2[i];
        // }
        // residual_connection<<<1,1>>>(x, s->xb2, dim);
        residual_connectionV2<<<dim/32,32,0,*stream>>>(x, s->xb2, dim);
        // ffn rmsnorm
        rmsnormV2<<<1,32,0,*stream>>>(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmulV2(handle, s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmulV2(handle, s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        swigluV2<<<hidden_dim/32,32,0,*stream>>>(s->hb, s->hb2, hidden_dim);


        // final matmul to get the output of the ffn
        matmulV2(handle, s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        residual_connectionV2<<<dim/32, 32,0,*stream>>>(x, s->xb, dim);

        // cudaStreamEndCapture(*stream, graph);
    
        // if ((*instance) == NULL) {
        //     printf("graphCreated: \n");
        //     err = cudaGraphInstantiate(instance, *graph, NULL, NULL, 0);
        //     if (err != cudaSuccess) {
        //         printf("cudaGraphInstantiate with error - %s\n", cudaGetErrorString(err));
        //     }
        // } else {
        //     cudaGraphExecUpdateResult updateResult_out;
            
        //     err = cudaGraphExecUpdate(*instance, *graph, NULL, &updateResult_out);
        //     // cudaGetErrorString(err);
        //     if (updateResult_out != cudaGraphExecUpdateSuccess) {
        //         printf("cudaGraphExecUpdate with error - %s\n", cudaGetErrorString(err));
        //         if ((*instance) != NULL) {
        //             cudaGraphExecDestroy(*instance);
        //         }
        //         printf("graph update failed with error - %d\n",updateResult_out);
        //         cudaGraphInstantiate(instance, *graph, NULL, NULL, 0);
        //     }
        // }

    // cudaGraphLaunch(*instance, *stream);
    // cudaStreamSynchronize(*stream);
    }

    // final rmsnorm
    rmsnormV2<<<1,32>>>(x, x, w->rms_final_weight, dim);
    // printf("xV2: \n");
    // print_vector<<<1,1>>>(x, dim);
    // cudaDeviceSynchronize();
    // exit(1);

    // classifier into logits
    matmulV2(handle, s->logits, x, w->wcls, p->dim, p->vocab_size);
    // printf("logitsV2: \n");
    // print_vector<<<1,1>>>(s->logits, p->vocab_size);
    // cudaDeviceSynchronize();
    // exit(1);

    // printf("x: \n");
    // print_vector<<<1,1>>>(x, dim);
    // cudaDeviceSynchronize();

    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char*)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    half prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;


// int sample_argmax(half* probabilities, int n) {
//     // return the index that has the highest probability
//     int max_i = 0;
//     half max_p = probabilities[0];
//     for (int i = 1; i < n; i++) {
//         if (probabilities[i] > max_p) {
//             max_i = i;
//             max_p = probabilities[i];
//         }
//     }
//     return max_i;
// }

__device__ __host__
void sample_argmax(int* max_i, half* probabilities, int n) {
    // return the index that has the highest probability
    *max_i = 0;
    half max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            *max_i = i;
            max_p = probabilities[i];
        }
    }
}

// int sample_mult(half* probabilities, int n, half coin) {
//     // sample index from probabilities (they must sum to 1!)
//     // coin is a random number in [0, 1), usually from random_f32()
//     half cdf = 0.0f;
//     for (int i = 0; i < n; i++) {
//         cdf += probabilities[i];
//         if (coin < cdf) {
//             return i;
//         }
//     }
//     return n - 1; // in case of rounding errors
// }

__device__
void sample_mult(int* next, half* probabilities, int n, half coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    half cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            *next = i;
            return;
        }
    }
    *next = n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

// int sample_topp(half* probabilities, int n, half topp, ProbIndex* probindex, half coin) {
//     // top-p sampling (or "nucleus sampling") samples from the smallest set of
//     // tokens that exceed probability topp. This way we never sample tokens that
//     // have very low probabilities and are less likely to go "off the rails".
//     // coin is a random number in [0, 1), usually from random_f32()

//     int n0 = 0;
//     // quicksort indices in descending order of probabilities
//     // values smaller than (1 - topp) / (n - 1) cannot be part of the result
//     // so for efficiency we crop these out as candidates before sorting
//     const half cutoff = ((half)1.0 - topp) / (half)(n - 1);
//     for (int i = 0; i < n; i++) {
//         if (probabilities[i] >= cutoff) {
//             probindex[n0].index = i;
//             probindex[n0].prob = probabilities[i];
//             n0++;
//         }
//     }
//     qsort(probindex, n0, sizeof(ProbIndex), compare);

//     // truncate the list where cumulative probability exceeds topp
//     half cumulative_prob = 0.0f;
//     int last_idx = n0 - 1; // in case of rounding errors consider all elements
//     for (int i = 0; i < n0; i++) {
//         cumulative_prob += probindex[i].prob;
//         if (cumulative_prob > topp) {
//             last_idx = i;
//             break; // we've exceeded topp by including last_idx
//         }
//     }

//     // sample from the truncated list
//     float r = coin * cumulative_prob;
//     float cdf = 0.0f;
//     for (int i = 0; i <= last_idx; i++) {
//         cdf += probindex[i].prob;
//         if (r < cdf) {
//             return probindex[i].index;
//         }
//     }
//     return probindex[last_idx].index; // in case of rounding errors
// }

__device__
void g_bubble_sort(ProbIndex *x, int size) {
    for (int i = 1; i < size; i++) {
        for (int j = 0; j < size - i; j++) {
            if (x[j].prob < x[j + 1].prob) {
                ProbIndex tmp = x[j];
                x[j] = x[j + 1];
                x[j+1] = tmp;
            }
        }
    }
}

__device__
void sample_topp(int *next, half* probabilities, int n, half topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const half cutoff = ((half)1.0 - topp) / (half)(n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }

    // qsort(probindex, n0, sizeof(ProbIndex), compare);
    g_bubble_sort(probindex, n0);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += __half2float(probindex[i].prob);
        if (cumulative_prob > __half2float(topp)) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }
    // for (int i = 0; i < n0; i++) {
    //     printf("sample_topp probindex[%d]=%f\n", probindex[i].index, __half2float(probindex[i].prob));
    // }
    // printf("sample_topp probindex n0=%d, last_idx=%d\n", n0, last_idx);
    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += __half2float(probindex[i].prob);
        if (r < cdf) {
            *next = probindex[i].index;
            // printf("sample_topp return i = %d, next=%d, r=%f, cdf=%f\n", i, *next, r, cdf);
            return;
        }
    }
    *next = probindex[last_idx].index; // in case of rounding errors
    // printf("sample_topp end next=%d\n", next);
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    // sampler->probindex = (ProbIndex*)malloc(sampler->vocab_size * sizeof(ProbIndex));
    void *device_probindex;
    cudaError_t err;
    err = cudaMalloc(&device_probindex, sampler->vocab_size * sizeof(ProbIndex));
    if (err != cudaSuccess) {
        printf("build_sampler error %d\n", err);
        exit(-1);
    }
    sampler->probindex = (ProbIndex*)device_probindex;
}

void free_sampler(Sampler* sampler) {
    cudaFree(sampler->probindex);
}

__device__ __host__
unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

__device__ __host__
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

__global__ void device_temperature(half* logits, int size, half temperature) {
    for (int i = 0; i < size; i++) {
        logits[i] /= temperature;
    }
}

__global__ 
void sample(int* next, Sampler* sampler, int vocab_size, ProbIndex* probindex,
                            float temperature, float topp, unsigned long long *rng_state, half* logits) {
    // printf("next=%d\n", *next);
    // sample the token given the logits and some hyperparameters
    if (temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        sample_argmax(next, logits, vocab_size);
    } else {
        // // apply the temperature to the logits
        for (int q=0; q<vocab_size; q++) { logits[q] /= __float2half(temperature); }
        
        // apply softmax to the logits to get the probabilities for next token
        device_softmax(logits, vocab_size);
        // printf("sample!\n");
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(rng_state);
        // we sample from this distribution to get the next token
        if (topp <= 0 || topp >= 1) {
            // simply sample from the predicted probability distribution
            sample_mult(next, logits, vocab_size, coin);
            printf("sample_mult next=%d\n", next);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            sample_topp(next, logits, vocab_size, topp, probindex, coin);
            
        }
        
    }
    // return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    printf("prompt_tokens: \n");
    for (int i = 0; i < strlen(prompt)+3; i++) {
        printf("%d , %d\n", i, prompt_tokens[i]);
    }
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    unsigned long long *device_rng_state;
    cudaMalloc(&device_rng_state, sizeof(unsigned long long));
    cudaMemcpy(device_rng_state, &sampler->rng_state, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    void* dnext;
    cudaMalloc(&dnext, sizeof(int));
    int token = prompt_tokens[0]; // kick off with the first token in the prompt

    int* dtokens;
    cudaMalloc((void **)&dtokens, steps*sizeof(int));
    cudaMemcpy((int*)dtokens, prompt_tokens, num_prompt_tokens*sizeof(int), cudaMemcpyHostToDevice);

    int pos = 0;     // position in the sequence
    // half* host_logits = (half*)calloc(sampler->vocab_size, sizeof(half));
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    bool graphCreated=false;

    // cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    // cudaStreamEndCapture(stream, &graph);
    // cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    // cudaGraphLaunch(instance, stream);
    // cudaStreamSynchronize(stream);
    cudaGraphNode_t blockDeviceNode[2];
    // create_global_graph(&graph_exec, &graph, &stream, &blockDeviceNode);

    while (pos < steps) {
        printf("pos=%d\n", pos);
        // forward the transformer to get logits for the next token

        half* logits = forward(&handle, &graph, &graph_exec, &stream, &graphCreated, blockDeviceNode, transformer, dtokens + pos, token, pos);
        // half* logits = forward(transformer, token, pos);
        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            // next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            // cudaMemcpy((void*)host_logits, logits, (sampler->vocab_size) * sizeof(half), cudaMemcpyDeviceToHost);
            // printf("logits=%p %p\n", logits, host_logits);
            // for (int i = 0; i < sampler->vocab_size; i++) {
            //     printf("%d=%f", i, __half2float(host_logits[i]));
            // }
            
    //             int vocab_size;
    // ProbIndex* probindex; // buffer used in top-p sampling
    // float temperature;
    // float topp;
            

            sample<<<1,1>>>(dtokens + pos + 1, sampler, sampler->vocab_size, sampler->probindex,
                            sampler->temperature, sampler->topp, device_rng_state, logits);
            // cudaDeviceSynchronize();
            // cudaMemcpy((void*)&next, dtokens + pos + 1, sizeof(int), cudaMemcpyDeviceToHost);
            // printf("next=%d\n", next);
        }
        pos++;

        // // data-dependent terminating condition: the BOS (=1) token delimits sequences
        // if (next == 1) { break; }

        // // print the token as string, decode it with the Tokenizer object
        // char* piece = decode(tokenizer, token, next);
        // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        // fflush(stdout);
        // token = next;
        // cudaMemcpy((int*)dtokens + pos, &token, sizeof(int), cudaMemcpyHostToDevice);

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // print_vector_int<<<1,1>>>(dtokens, steps);
    // cudaDeviceSynchronize();

    int *htokens = (int*)malloc(steps*sizeof(int));
    cudaMemcpy((int*)htokens, (int*)dtokens, steps*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 1; i < steps; i++) {
        if (htokens[i] == 1) { break; }
        char* piece = decode(tokenizer, htokens[i-1], htokens[i]);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        
        // printf("htokens[%d]=%d\n",i, htokens[i]);
    }
    printf("\n");
    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
    cublasDestroy(handle);
    // free(host_logits);
    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // // buffers for reading the system prompt and user prompt from stdin
    // // you'll notice they are soomewhat haphazardly and unsafely set atm
    // char system_prompt[512];
    // char user_prompt[512];
    // char rendered_prompt[1152];
    // int num_prompt_tokens = 0;
    // int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    // int user_idx;

    // // start the main loop
    // int8_t user_turn = 1; // user starts
    // int next;        // will store the next token in the sequence
    // int token;       // stores the current token to feed into the transformer
    // int prev_token;
    // int pos = 0;     // position in the sequence
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // while (pos < steps) {

    //     // when it is the user's turn to contribute tokens to the dialog...
    //     if (user_turn) {
    //         // get the (optional) system prompt at position 0
    //         if (pos == 0) {
    //             // at position 0, the user can also contribute a system prompt
    //             if (cli_system_prompt == NULL) {
    //                 // system prompt was not passed in, attempt to get it from stdin
    //                 read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
    //             } else {
    //                 // system prompt was passed in, use it
    //                 strcpy(system_prompt, cli_system_prompt);
    //             }
    //         }
    //         // get the user prompt
    //         if (pos == 0 && cli_user_prompt != NULL) {
    //             // user prompt for position 0 was passed in, use it
    //             strcpy(user_prompt, cli_user_prompt);
    //         } else {
    //             // otherwise get user prompt from stdin
    //             read_stdin("User: ", user_prompt, sizeof(user_prompt));
    //         }
    //         // render user/system prompts into the Llama 2 Chat schema
    //         if (pos == 0 && system_prompt[0] != '\0') {
    //             char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
    //             sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
    //         } else {
    //             char user_template[] = "[INST] %s [/INST]";
    //             sprintf(rendered_prompt, user_template, user_prompt);
    //         }
    //         // encode the rendered prompt into tokens
    //         encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    //         user_idx = 0; // reset the user index
    //         user_turn = 0;
    //         printf("Assistant: ");
    //     }

    //     // determine the token to pass into the transformer next
    //     if (user_idx < num_prompt_tokens) {
    //         // if we are still processing the input prompt, force the next prompt token
    //         token = prompt_tokens[user_idx++];
    //     } else {
    //         // otherwise use the next token sampled from previous turn
    //         token = next;
    //     }
    //     // EOS (=2) token ends the Assistant turn
    //     if (token == 2) { user_turn = 1; }

    //     // forward the transformer to get logits for the next token
    //     half* logits = forward(&handle, transformer, token, pos);
    //     sample<<<1,1>>>(&next, sampler, logits);
    //     pos++;

    //     if (user_idx >= num_prompt_tokens && next != 2) {
    //         // the Assistant is responding, so print its output
    //         char* piece = decode(tokenizer, token, next);
    //         safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    //         fflush(stdout);
    //     }
    //     if (next == 2) { printf("\n"); }
    // }
    // printf("\n");
    // cublasDestroy(handle);
    // free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    // Sampler *device_sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
    // cudaMemcpy(device_sampler, &sampler, sizeof(Sampler), cudaMemcpyHostToDevice);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    printf("run end!\n");
    return 0;
}
#endif