// Teacher-forced KL(P||Q) between two GGUF models over stored token sequences. Driven by
// qtrace/kl_gguf.py, which documents the build command.
//
// Usage: kl_score P.gguf Q.gguf JOBS [n_batch]
//
// JOBS has one run per line, "<tokens> <idx> <out>" (paths without spaces). tokens holds int32 token
// ids; idx holds sorted, unique int32 positions to score, where the logit at position i predicts
// token i + 1. out receives one float32 row per idx entry: kl, flip, entropy_p, nll_p, nll_q, the
// fields of compare() in qtrace/kl.py. Both models load once and every run reuses them, because a
// load costs more than scoring a short run.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

static void die(const std::string & msg) {
    fprintf(stderr, "kl_score: %s\n", msg.c_str());
    exit(1);
}

static std::vector<int32_t> read_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) die("cannot read " + path);
    const std::streamoff n = f.tellg();
    if (n % 4) die(path + " is not a whole number of int32 values");
    std::vector<int32_t> v(n / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), n);
    if (!f) die("short read on " + path);
    return v;
}

struct Model {
    llama_model * model;
    llama_context * ctx;
};

static Model load(const char * path, uint32_t n_ctx, uint32_t n_batch) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;  // all layers
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) die(std::string("cannot load ") + path);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.n_seq_max = 1;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die(std::string("cannot create a context for ") + path);
    return {model, ctx};
}

// The compare() fields for one position. Sums over the vocabulary run in double.
static void row_stats(const float * a, const float * b, int n_vocab, int y, float * out) {
    int ia = 0, ib = 0;  // first index of the max, like torch.argmax
    for (int i = 1; i < n_vocab; ++i) {
        if (a[i] > a[ia]) ia = i;
        if (b[i] > b[ib]) ib = i;
    }
    double sa = 0, sb = 0;
    for (int i = 0; i < n_vocab; ++i) {
        sa += std::exp(double(a[i]) - a[ia]);
        sb += std::exp(double(b[i]) - b[ib]);
    }
    const double za = a[ia] + std::log(sa), zb = b[ib] + std::log(sb);  // log-sum-exp
    double kl = 0, ent = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const double la = a[i] - za, lb = b[i] - zb, p = std::exp(la);
        kl += p * (la - lb);
        ent -= p * la;
    }
    out[0] = float(kl);
    out[1] = ia != ib;
    out[2] = float(ent);
    out[3] = float(za - a[y]);
    out[4] = float(zb - b[y]);
}

static std::vector<float> score(Model & P, Model & Q, int n_vocab, int n_batch, const std::vector<int32_t> & tok,
                                const std::vector<int32_t> & idx) {
    const int n = int(tok.size());
    for (size_t k = 0; k < idx.size(); ++k) {
        if (idx[k] < 0 || idx[k] >= n - 1 || (k && idx[k] <= idx[k - 1])) {
            die("idx must be sorted, unique and in [0, n_tokens - 1)");
        }
    }
    for (int32_t t : tok) {
        if (t < 0 || t >= n_vocab) die("token id " + std::to_string(t) + " is outside the vocabulary");
    }
    std::vector<float> out(idx.size() * 5);
    if (idx.empty()) return out;

    llama_memory_clear(llama_get_memory(P.ctx), true);
    llama_memory_clear(llama_get_memory(Q.ctx), true);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    const unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    size_t k = 0;  // next idx entry
    for (int s = 0; s < n && k < idx.size(); s += n_batch) {
        const int m = std::min(n_batch, n - s);
        const size_t k0 = k;
        batch.n_tokens = m;
        for (int j = 0; j < m; ++j) {
            batch.token[j] = tok[s + j];
            batch.pos[j] = s + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = k < idx.size() && idx[k] == s + j;
            if (batch.logits[j]) ++k;
        }
        if (llama_decode(P.ctx, batch) != 0 || llama_decode(Q.ctx, batch) != 0) die("llama_decode failed");
        std::vector<const float *> lp(k - k0), lq(k - k0);
        for (size_t r = k0; r < k; ++r) {
            lp[r - k0] = llama_get_logits_ith(P.ctx, idx[r] - s);
            lq[r - k0] = llama_get_logits_ith(Q.ctx, idx[r] - s);
            if (!lp[r - k0] || !lq[r - k0]) die("missing logits");
        }
        std::vector<std::thread> pool;
        for (unsigned w = 0; w < n_threads; ++w) {
            pool.emplace_back([&, w] {
                for (size_t r = k0 + w; r < k; r += n_threads) {
                    row_stats(lp[r - k0], lq[r - k0], n_vocab, tok[idx[r] + 1], &out[r * 5]);
                }
            });
        }
        for (auto & t : pool) t.join();
    }
    llama_batch_free(batch);
    return out;
}

static void quiet(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) die("usage: kl_score P.gguf Q.gguf JOBS [n_batch]");
    const int n_batch = argc == 5 ? atoi(argv[4]) : 512;
    if (n_batch < 1) die("n_batch must be a positive integer");

    struct Job {
        std::string tok, idx, out;
    };
    std::vector<Job> jobs;
    std::ifstream f(argv[3]);
    if (!f) die(std::string("cannot read ") + argv[3]);
    for (Job j; f >> j.tok >> j.idx >> j.out;) jobs.push_back(j);
    if (jobs.empty()) die("no jobs in " + std::string(argv[3]));

    uintmax_t n_ctx = 1;  // the longest run sets the context size
    for (const Job & j : jobs) n_ctx = std::max(n_ctx, std::filesystem::file_size(j.tok) / 4);

    llama_log_set(quiet, nullptr);
    llama_backend_init();
    Model P = load(argv[1], uint32_t(n_ctx), uint32_t(n_batch));
    Model Q = load(argv[2], uint32_t(n_ctx), uint32_t(n_batch));
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(P.model));
    if (n_vocab != llama_vocab_n_tokens(llama_model_get_vocab(Q.model))) die("the two models have different vocabularies");

    for (const Job & j : jobs) {
        const std::vector<float> out = score(P, Q, n_vocab, n_batch, read_i32(j.tok), read_i32(j.idx));
        std::ofstream o(j.out, std::ios::binary);
        o.write(reinterpret_cast<const char *>(out.data()), std::streamsize(out.size() * sizeof(float)));
        if (!o) die("cannot write " + j.out);
    }

    llama_free(P.ctx);
    llama_free(Q.ctx);
    llama_model_free(P.model);
    llama_model_free(Q.model);
    llama_backend_free();
    return 0;
}
