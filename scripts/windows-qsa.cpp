// The runner extracts the current production metadata function into qsa-input-test.inc.
// Cache adapters avoid loading a model; all tensor storage in this test is on the host.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "llama-batch.h"
#include "prefix.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#undef GGML_ASSERT
#define GGML_ASSERT(x) do { if (!(x)) { throw std::runtime_error(#x); } } while (0)
#define ggml_backend_buffer_is_host(buffer) true

struct test_cache {
    llama_kv_cells cells;
    const llama_kv_cells & get_cells(llama_seq_id) const { return cells; }
};

struct llama_memory_hybrid_idx {
    test_cache cache;
    const test_cache * get_mem_idx() const { return &cache; }
    bool qsa_metadata(ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, const llama_ubatch &, uint32_t) const { return false; }
    void set_input_qsa_impl(ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, const llama_ubatch *, uint32_t, bool) const;
};

#include "qsa-input-test.inc"
#include "qsa-visibility-test.inc"

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct cell_input {
    llama_pos pos;
    llama_seq_id seq;
};

// Check metadata against a logical-position reference, independent of physical cell order.
static void run_case(const char * name, const std::vector<cell_input> & input,
        const std::vector<cell_input> & queries, uint32_t n_kv, bool contiguous, bool blocks, bool compact = false) {
    constexpr int r = 4;
    const int n_blocks = (n_kv+r-1)/r;
    const size_t nt = queries.size();
    llama_memory_hybrid_idx mem;
    mem.cache.cells.resize(n_kv);
    using key = std::pair<llama_seq_id, llama_pos>;
    std::map<key, std::vector<int>> reference;
    for (size_t j = 0; j < input.size(); ++j) {
        mem.cache.cells.pos_set(j, input[j].pos);
        mem.cache.cells.seq_add(j, input[j].seq);
        reference[{input[j].seq, input[j].pos/r}].push_back(j);
    }
    check(qsa_contiguous_sequences(mem.cache.cells, n_kv) == contiguous, "incorrect contiguous-cache classification");
    const bool blk_bias = blocks || contiguous;

    std::vector<llama_pos> positions;
    std::vector<llama_seq_id> ids;
    for (auto q : queries) { positions.push_back(q.pos); ids.push_back(q.seq); }
    std::vector<llama_seq_id *> seqs;
    for (auto & id : ids) { seqs.push_back(&id); }
    llama_ubatch u = {};
    u.n_tokens = nt; u.n_pos = 1; u.pos = positions.data(); u.seq_id = seqs.data();

    // One guard element follows each output to detect writes past its declared dimensions.
    constexpr int32_t guard = 0x345678;
    std::vector<int32_t> cell_data(n_kv+1, guard), block_data(r*n_blocks+1, guard), pos_data(4*n_blocks+1, guard);
    std::vector<int32_t> tail_data((r-1)*nt+1, guard), limits(n_blocks+nt+1, guard);
    std::vector<float> bias_data((blk_bias ? n_blocks : n_kv)*nt+1, float(guard));
    ggml_tensor cell = {}, block = {}, pos = {}, bias = {}, tail = {};
    cell.ne[0] = n_kv; cell.ne[1] = 1; cell.data = cell_data.data();
    block.data = block_data.data();
    pos.ne[0] = 4*n_blocks; pos.data = pos_data.data();
    bias.type = compact ? GGML_TYPE_I32 : GGML_TYPE_F32;
    bias.ne[0] = compact ? n_blocks+nt : (blk_bias ? n_blocks : n_kv);
    bias.ne[1] = compact ? 1 : nt; bias.ne[2] = bias.ne[3] = 1;
    bias.data = compact ? (void *) limits.data() : (void *) bias_data.data();
    tail.ne[0] = r-1; tail.ne[1] = nt; tail.ne[2] = tail.ne[3] = 1; tail.data = tail_data.data();
    mem.set_input_qsa_impl(&cell, &block, &pos, &bias, blocks ? &tail : nullptr, &u, r, blk_bias);

    check(cell_data.back() == guard && block_data.back() == guard && pos_data.back() == guard &&
            tail_data.back() == guard && limits.back() == guard && bias_data.back() == float(guard), "metadata buffer overrun");

    std::set<int> pooled;
    std::set<key> complete;
    for (const auto & entry : reference) {
        std::set<llama_pos> positions;
        for (int j : entry.second) { positions.insert(input[j].pos); }
        if (positions.size() == r) { complete.insert(entry.first); }
    }
    for (size_t b = 0; b < complete.size(); ++b) {
        const int first = block_data[r*b];
        check(first >= 0 && size_t(first) < input.size(), "invalid pooled cell");
        const key k = {input[first].seq, input[first].pos/r};
        check(complete.count(k) != 0, "pooled an incomplete block");
        check(pos_data[b] == k.second*r, "lost logical block position");
        for (int slot = 0; slot < r; ++slot) {
            const int j = block_data[r*b+slot];
            check(j >= 0 && size_t(j) < input.size(), "invalid block member");
            check(input[j].seq == k.first && input[j].pos == k.second*r+slot, "mixed block positions or sequences");
            check(pooled.insert(j).second, "pooled a cell twice");
            if (!blocks) { check(cell_data[j] == int(b), "incorrect cell-to-block mapping"); }
        }
    }

    for (size_t i = 0; i < nt; ++i) {
        const auto q = queries[i];
        const llama_pos tail_start = (q.pos+1)/r*r;
        std::set<int> expected_visible, actual_visible;
        for (size_t j = 0; j < input.size(); ++j) {
            const auto c = input[j];
            const bool visible = c.seq == q.seq && c.pos <= q.pos &&
                    (c.pos >= tail_start || complete.count({c.seq, c.pos/r}));
            if (visible) { expected_visible.insert(j); }
        }
        if (blocks) {
            if (compact) { check(limits[n_blocks+i] == tail_start, "incorrect compact query limit"); }
            for (size_t b = 0; b < complete.size(); ++b) {
                const int j = block_data[r*b];
                const bool visible = compact ? limits[b] < limits[n_blocks+i] : std::isfinite(bias_data[i*n_blocks+b]);
                if (visible) {
                    check(input[j].seq == q.seq && input[j].pos+r-1 <= q.pos, "visible foreign or future block");
                    for (int slot = 0; slot < r; ++slot) { actual_visible.insert(block_data[r*b+slot]); }
                }
            }
            for (int slot = 0; slot < r-1; ++slot) {
                const int j = tail_data[i*(r-1)+slot];
                if (j >= 0) { actual_visible.insert(j); }
            }
        } else {
            for (uint32_t j = 0; j < n_kv; ++j) {
                check(cell_data[j] >= 0 && cell_data[j] < n_blocks, "out-of-range expanded score index");
                float value = bias_data[i*(blk_bias ? n_blocks : n_kv)+(blk_bias ? cell_data[j] : j)];
                if (blk_bias && (j >= input.size() || input[j].seq != q.seq || input[j].pos > q.pos)) { value = -INFINITY; }
                if (std::isfinite(value)) { actual_visible.insert(j); }
                if (expected_visible.count(j)) {
                    check(value == (input[j].pos >= tail_start ? 1e9f : 0.0f), "incorrect tail priority");
                }
            }
        }
        check(actual_visible == expected_visible, "causal cell visibility differs from reference");
    }
    printf("PASS: %s (%s)\n", name, compact ? "compact blocks" : (blocks ? "explicit blocks" : (blk_bias ? "block bias" : "cell bias")));
}

static void check_shared_input_views() {
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    check(ctx != nullptr, "view test context allocation failed");
    std::vector<ggml_tensor *> views;
    auto * bias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 10, 1);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 8, 10, 1, 1);
    for (int layer = 0; layer < 12; ++layer) {
        for (int first = 0; first < 10; first += 4) {
            const int count = std::min(4, 10-first);
            auto * score = qwen4exp_shared_input_view(&views, ggml_view_3d(ctx, bias, 8, count, 1,
                    bias->nb[1], bias->nb[2], first*bias->nb[1]));
            auto * attn = qwen4exp_shared_input_view(&views, ggml_view_4d(ctx, bias, 8, count, 1, 1,
                    bias->nb[1], bias->nb[2], bias->nb[2], first*bias->nb[1]));
            check(score == attn, "equivalent bias slices were not shared");
            for (int use = 0; use < 2; ++use) {
                qwen4exp_shared_input_view(&views, ggml_view_4d(ctx, mask, 8, count, 1, 1,
                        mask->nb[1], mask->nb[2], mask->nb[3], first*mask->nb[1]));
            }
        }
    }
    check(views.size() == 6, "layer count multiplied input slices");
    auto * other = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 10, 1);
    auto * different_source = qwen4exp_shared_input_view(&views, ggml_view_3d(ctx, other, 8, 4, 1, other->nb[1], other->nb[2], 0));
    auto * different_stride = qwen4exp_shared_input_view(&views, ggml_view_3d(ctx, bias, 8, 4, 1, 2*bias->nb[1], bias->nb[2], 0));
    check(different_source != views[0] && different_stride != views[0] && views.size() == 8, "different source or strides were aliased");
    auto * unshared = ggml_view_3d(ctx, bias, 8, 4, 1, bias->nb[1], bias->nb[2], 0);
    check(qwen4exp_shared_input_view(nullptr, unshared) == unshared, "disabled sharing changed the input view");
    ggml_free(ctx);
    printf("PASS: shared input slices preserve source, offset, shape, and strides\n");
}

static void check_final_mask(ggml_type type, ggml_backend_t backend) {
    ggml_context * ctx = ggml_init({1024*1024, nullptr, true});
    check(ctx != nullptr, "mask test context allocation failed");
    auto * bias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 2);
    auto * raw_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 2, 1, 2);
    float values[64];
    for (int i = 0; i < 64; ++i) { values[i] = i%3 == 0 ? -INFINITY : (i%3 == 1 ? 1e9f : 0.0f); }
    float original[32];
    for (int i = 0; i < 32; ++i) { original[i] = i%5 == 0 ? -INFINITY : -2.0f; }
    auto * mask = type == GGML_TYPE_F32 ? raw_mask : ggml_cast(ctx, raw_mask, type);
    std::vector<ggml_tensor *> views;
    auto * result = qwen4exp_apply_cell_visibility(ctx, mask, bias, 1, &views);
    if (result->type != GGML_TYPE_F32) { result = ggml_cast(ctx, result, GGML_TYPE_F32); }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "mask buffer allocation failed");
    ggml_backend_tensor_set(bias, values, 0, sizeof(values));
    ggml_backend_tensor_set(raw_mask, original, 0, sizeof(original));
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "mask graph failed");
    float output[32];
    ggml_backend_tensor_get(result, output, 0, sizeof(output));
    for (int stream = 0; stream < 2; ++stream) {
        for (int q = 0; q < 2; ++q) {
            for (int j = 0; j < 8; ++j) {
                const int out = stream*16+q*8+j;
                const int in = stream*32+(q+1)*8+j;
                const float expected = std::isfinite(values[in]) ? original[out] : -INFINITY;
                if (output[out] != expected) { fprintf(stderr, "mask mismatch: type=%s stream=%d q=%d cell=%d actual=%g expected=%g\n", ggml_type_name(type), stream, q, j, output[out], expected); }
                check(output[out] == expected, "final mask lost exclusions, stream offset, or changed attention logits");
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    printf("PASS: final attention mask with %s storage and two streams on %s\n", ggml_type_name(type), ggml_backend_name(backend));
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--gpu") {
            ggml_backend_load_all();
            auto * device = ggml_backend_dev_by_name("ROCm0");
            check(device != nullptr, "ROCm0 unavailable");
            auto * backend = ggml_backend_dev_init(device, nullptr);
            check(backend != nullptr, "ROCm0 initialization failed");
            check_final_mask(GGML_TYPE_F32, backend);
            check_final_mask(GGML_TYPE_F16, backend);
            ggml_backend_free(backend);
            return 0;
        }
        std::vector<cell_input> logged;
        check_shared_input_views();
        for (int j = 0; j < 13526; ++j) { logged.push_back({j < 13522 ? j : j+45, 1}); }
        const std::vector<cell_input> queries = {{13567,1},{13568,1},{13569,1},{13570,1}};
        run_case("reported screenshot layout", logged, queries, 13568, false, false);
        run_case("reported screenshot layout", logged, queries, 13568, false, true);
        run_case("reported screenshot layout", logged, queries, 13568, false, true, true);

        std::vector<cell_input> gaps = {{0,1},{1,1},{8,1},{9,1},{10,1},{11,1},
                {1000000,1},{1000001,1},{1000002,1},{1000003,1},{1000004,1},{1000005,1}};
        std::reverse(gaps.begin(), gaps.end());
        const std::vector<cell_input> far = {{9,1},{1000002,1},{1000003,1},{1000005,1}};
        run_case("large gaps and physical permutation", gaps, far, 16, false, false);
        run_case("large gaps and physical permutation", gaps, far, 16, false, true);
        run_case("large gaps and physical permutation", gaps, far, 16, false, true, true);

        std::vector<cell_input> two = {{0,0},{0,1},{1,0},{1,1},{2,0},{2,1},{3,0},{3,1},{4,0},{4,1}};
        const std::vector<cell_input> both = {{2,0},{3,1},{4,0},{4,1}};
        run_case("two contiguous sequences", two, both, 16, true, false);
        run_case("two contiguous sequences", two, both, 16, true, true);
        for (auto & c : two) { if (c.seq == 1) { c.pos += 100; } }
        run_case("two sequences with a missing prefix", two, {{3,0},{103,1},{4,0},{104,1}}, 16, false, false);
        run_case("two sequences with a missing prefix", two, {{3,0},{103,1},{4,0},{104,1}}, 16, false, true);

        test_cache cache;
        cache.cells.resize(8);
        cache.cells.pos_set(0, 0); cache.cells.seq_add(0, 0); cache.cells.seq_add(0, 1);
        check(!qsa_contiguous_sequences(cache.cells, 8), "shared cell accepted by conservative block-bias guard");
        cache.cells.reset();
        for (int j = 0; j < 2; ++j) { cache.cells.pos_set(j, 0); cache.cells.seq_add(j, 0); }
        check(!qsa_contiguous_sequences(cache.cells, 8), "duplicate position accepted by block-bias guard");
        printf("PASS: shared and repeated-position layouts require cell masking\n");
        auto * backend = ggml_backend_cpu_init();
        check(backend != nullptr, "CPU backend unavailable");
        check_final_mask(GGML_TYPE_F32, backend);
        check_final_mask(GGML_TYPE_F16, backend);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
