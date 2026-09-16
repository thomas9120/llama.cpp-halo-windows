// The runner extracts production k-pool code; cache adapters avoid loading a model.
#include "llama-kv-cache-kpool.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#undef GGML_ASSERT
#define GGML_ASSERT(x) do { if (!(x)) { throw std::runtime_error(#x); } } while (0)
#define ggml_backend_buffer_is_host(buffer) true

struct llama_kv_cache {
    llama_kv_cells cells;
    mutable bool dirty = false;
    const llama_kv_cells & get_cells(llama_seq_id) const { return cells; }
    uint32_t get_size() const { return cells.size(); }
    bool get_kpool_dirty() const { return dirty; }
    void clear_kpool_dirty() const { dirty = false; }
};

struct llama_kv_cache_context {
    llama_kv_cache * kv;
    uint32_t n_kv = 1280;
    mutable int writes = 0;
    const llama_kv_cache * get_kv() const { return kv; }
    uint32_t get_n_kv() const { return n_kv; }
    uint32_t get_n_stream() const { return 1; }
    uint32_t get_strm(uint32_t s) const { return s; }
    void set_input_k_idxs(ggml_tensor *, const llama_ubatch *) const { ++writes; }
};

struct llama_memory_hybrid_context : llama_memory_context_i {
    const llama_kv_cache_context * attn;
    const llama_kv_cache_context * idx;
    const llama_kv_cache_context * get_attn() const { return attn; }
    const llama_kv_cache_context * get_idx() const { return idx; }
    bool next() override { return false; }
    bool apply() override { return true; }
    const llama_ubatch & get_ubatch() const override { throw std::runtime_error("unused"); }
    llama_memory_status get_status() const override { return LLAMA_MEMORY_STATUS_SUCCESS; }
};

#include "kpool-input-test.inc"

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static void check_shared_prefix(ggml_type mask_type, int nseq, int nt, bool rebuild) {
    constexpr int n_kv = 1280, prefix = 1024, r = 4;
    llama_kv_cache kv;
    kv.cells.resize(n_kv);
    for (int j = 0; j < prefix; ++j) {
        kv.cells.pos_set(j, j);
        for (int s = 0; s < nseq; ++s) { kv.cells.seq_add(j, s); }
    }
    std::vector<llama_pos> positions(nseq*nt);
    std::vector<llama_seq_id> ids(nseq), query_ids(nseq*nt);
    std::vector<llama_seq_id *> seqs(nseq*nt);
    for (int s = 0; s < nseq; ++s) {
        ids[s] = s;
        for (int t = 0; t < nt; ++t) {
            const int i = s*nt+t;
            positions[i] = prefix+t;
            query_ids[i] = s;
            seqs[i] = &query_ids[i];
            kv.cells.pos_set(prefix+i, prefix+t);
            kv.cells.seq_add(prefix+i, s);
        }
    }
    llama_ubatch u = {};
    u.n_tokens = nseq*nt; u.n_seqs = nseq; u.n_seqs_unq = nseq; u.n_seq_tokens = nt;
    u.n_pos = 1; u.pos = positions.data(); u.seq_id = seqs.data(); u.seq_id_unq = ids.data();
    auto * ctx = ggml_init({8*1024*1024, nullptr, false});
    check(ctx != nullptr, "tensor allocation failed");
    const int np = llama_kpool_n_pools(n_kv, r, nseq);
    const int nn = rebuild ? np : nseq*nt/r+nseq;
    auto * pc = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r*np, 1);
    auto * pb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, np, nseq*nt, 1);
    auto * sm = ggml_new_tensor_4d(ctx, mask_type, n_kv, nseq*nt, 1, 1);
    auto * cm = ggml_new_tensor_4d(ctx, mask_type, n_kv, nseq*nt, 1, 1);
    auto * reps = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, np, 1);
    auto * nc = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r*nn, 1);
    auto * nr = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nn);
    const uint32_t stream = 0;
    llama_kv_cache_set_input_kpool(&kv, nullptr, pc, nullptr, pb, sm, cm, reps, nc, nr, &stream, n_kv, rebuild, &u, r);
    for (int i = 0; i < nseq*nt; ++i) {
        int complete = 0;
        for (int p = 0; p < np; ++p) {
            if (!std::isfinite(((float *) pb->data)[i*np+p])) { continue; }
            ++complete;
            for (int k = 0; k < r; ++k) {
                const int j = ((int32_t *) pc->data)[p*r+k];
                check(kv.cells.seq_has(j, query_ids[i]) && kv.cells.pos_get(j) <= positions[i], "foreign or future pool visible");
            }
        }
        check(complete == (positions[i]+1)/r, "shared prefix lost complete pools");
        for (int j = 0; j < n_kv; ++j) {
            const bool expected = !kv.cells.is_empty(j) && kv.cells.seq_has(j, query_ids[i]) && kv.cells.pos_get(j) <= positions[i];
            const float value = mask_type == GGML_TYPE_F32 ? ((float *) cm->data)[i*n_kv+j] : ggml_fp16_to_fp32(((ggml_fp16_t *) cm->data)[i*n_kv+j]);
            check(std::isfinite(value) == expected, "candidate mask lost context or exposed another sequence");
        }
    }
    ggml_free(ctx);
    printf("PASS: GLM shared prefix, %d sequences, %d tokens/sequence, %s, rebuild=%d\n", nseq, nt, ggml_type_name(mask_type), rebuild);
}

static void check_reuse(bool unified) {
    llama_kv_cache kv;
    kv.cells.resize(1280);
    llama_kv_cache_context old_ctx{&kv}, new_ctx{&kv};
    llama_memory_hybrid_context memory;
    memory.attn = memory.idx = &new_ctx;
    llm_graph_params params = {};
    params.mctx = &memory;
    params.cparams.kv_unified = unified;
    params.ubatch.n_tokens = params.ubatch.n_seqs_unq = 2;
    auto * ctx = ggml_init({4*1024*1024, nullptr, false});
    check(ctx != nullptr, "tensor allocation failed");
    llm_graph_input_kpool input(&old_ctx, &old_ctx, 4);
    input.k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 2);
    llm_graph_input_i & base = input;
    check(base.can_reuse(params), "dense indexer graph cannot be reused");
    input.set_input(&params.ubatch);
    check(old_ctx.writes == 0 && new_ctx.writes == 1, "reused graph kept the old cache context");
    const int ns = unified ? 1 : 2, nt = 2/ns, np = llama_kpool_n_pools(1280, 4, 2/ns);
    input.pool_cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4*np, ns);
    input.pool_bias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, np, nt, ns);
    input.sel_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1280, nt, 1, ns);
    input.n_new_max = nt/4+2/ns;
    check(base.can_reuse(params), "stable sparse graph cannot be reused");
    ++new_ctx.n_kv;
    check(!base.can_reuse(params), "changed KV mask shape reused");
    --new_ctx.n_kv;
    ++params.ubatch.n_tokens;
    check(!base.can_reuse(params), "changed token count reused");
    --params.ubatch.n_tokens;
    params.ubatch.n_seqs_unq = 1;
    check(!base.can_reuse(params), "changed sequence layout reused");
    params.ubatch.n_seqs_unq = 2;
    kv.dirty = true;
    check(!base.can_reuse(params), "dirty cache reused without rebuilding pools");
    input.rebuild = true;
    input.n_new_max = np;
    check(base.can_reuse(params), "matching rebuild graph rejected");
    kv.dirty = false;
    check(!base.can_reuse(params), "rebuild graph reused after dirty flag cleared");
    memory.idx = nullptr;
    check(!base.can_reuse(params), "missing indexer accepted");
    ggml_free(ctx);
    printf("PASS: GLM graph reuse and invalidation, unified=%d\n", unified);
}

int main() {
    try {
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            for (int nseq : {1, 2, 3}) {
                for (int nt : {1, 4}) {
                    for (bool rebuild : {false, true}) { check_shared_prefix(type, nseq, nt, rebuild); }
                }
            }
        }
        check_reuse(false);
        check_reuse(true);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
