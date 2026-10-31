// llama-dump-hidden: runs the target model over rendered conversations and stores what draft-model training needs.
//
// MTP mode (default, unchanged): the token ids, a mask of trainable tokens (the model's own output), and the target's
// final hidden state (result_norm, which is exactly the MTP head's h input in this fork) at every position the head
// trains on.
//
// DFlash mode (env DUMP_LAYERS set): per sequence the token ids, the train mask, and
//   - the INPUT residual of each listed target layer (llama_set_embeddings_layer_inp, the exact tensor that
//     common/speculative.cpp feeds DFlash in production; ids in the GGUF dflash.target_layers convention, i.e.
//     "6,20,34,48,62" = HF hidden_states[6..62] = outputs of decoder blocks 5..61) at every position that some anchor
//     can see through the drafter's sliding window;
//   - result_norm at every position whose next-token distribution is a distillation target (Swift's own sampler
//     distribution is computed from it in training with the GGUF's output matrix, as for MTP).
//
// usage: llama-dump-hidden <normal model/context args, e.g. -m x.gguf -ngl 999 -fa on -c 32768 -b 2048 -ub 512>
//   env DUMP_IN     = input JSONL, one object per line: {"id": str, "text": str, "train": [[byte_start, byte_end], ...]}
//   env DUMP_OUT    = output path prefix: writes <prefix>.bin (records) and <prefix>.idx.jsonl (one line per record)
//   env DUMP_AHEAD  = MTP mode: positions kept before each trainable token (default 4: the recursive draft steps need them)
//   env DUMP_LAYERS = DFlash mode: comma-separated target layer ids whose input is dumped
//   env DUMP_WINDOW = DFlash mode: a feature row is kept if an anchor lies within this many positions after it
//                     (default 2047: the drafter's 2048 sliding window; a block query at anchor+i sees keys p with
//                     anchor+i - p < 2048 and p < anchor, so the union over i is p >= anchor - 2047)
//
// MTP record layout in .bin (little endian), per sequence:
//   int32 n_tokens, int32 n_keep, int32 tokens[n_tokens], uint8 train[n_tokens], uint8 keep[n_tokens],
//   fp16 hidden[n_keep][n_embd]   (rows for the positions with keep=1, in order)
//
// DFlash record layout in .bin (little endian), per sequence:
//   int32 magic 0x32464644 ("DDF2"), int32 n_tokens, int32 n_feat, int32 n_h, int32 n_layers, int32 n_embd,
//   int32 tokens[n_tokens], uint8 train[n_tokens], uint8 keep_f[n_tokens], uint8 keep_h[n_tokens],
//   bf16 feat[n_feat][n_layers * n_embd]   (positions with keep_f=1, in order; inside a row layer k occupies
//                                           [k*n_embd, (k+1)*n_embd), the same layout as features_buf in
//                                           common/speculative.cpp),
//   fp16 h[n_h][n_embd]                    (result_norm rows for the positions with keep_h=1, in order)
//   keep_h[j] = train[j] && train[j+1]: h[j] gives the target's distribution of token j+1, needed when j+1 is a
//   drafted position of a block whose anchor and drafts are model output.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-ext.h" // staging API: llama_set_embeddings_layer_inp / llama_get_embeddings_layer_inp (as speculative.cpp)
#include "ggml.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static std::string env_or(const char * name, const std::string & def) {
    const char * v = std::getenv(name);
    return v && *v ? std::string(v) : def;
}

static std::vector<int> parse_layers(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::atoi(item.c_str()));
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }
    const std::string in_path  = env_or("DUMP_IN", "");
    const std::string out_path = env_or("DUMP_OUT", "");
    const int ahead = std::atoi(env_or("DUMP_AHEAD", "4").c_str());
    const std::vector<int> layers = parse_layers(env_or("DUMP_LAYERS", ""));
    const bool dflash = !layers.empty();
    const int window = std::atoi(env_or("DUMP_WINDOW", "2047").c_str());
    if (in_path.empty() || out_path.empty()) {
        LOG_ERR("set DUMP_IN and DUMP_OUT\n");
        return 1;
    }
    params.embedding = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.n_parallel = 1;

    llama_backend_init();
    llama_numa_init(params.numa);
    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("unable to load model\n");
        return 1;
    }
    const int n_embd  = llama_model_n_embd(model);
    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = llama_n_batch(ctx);
    const int n_layer = llama_model_n_layer(model);
    const int n_feat_row = (int) layers.size() * n_embd;

    if (dflash) {
        for (const int lid : layers) {
            if (lid < 0 || lid >= n_layer) {
                LOG_ERR("DUMP_LAYERS: layer %d out of range [0, %d)\n", lid, n_layer);
                return 1;
            }
            // same call as common_speculative_impl_draft_dflash's constructor on ctx_tgt
            llama_set_embeddings_layer_inp(ctx, (uint32_t) lid, true);
        }
        LOG_INF("DFlash mode: %zu layers (%d floats per feature row), window %d\n", layers.size(), n_feat_row, window);
    }

    std::ifstream in(in_path, std::ios::binary);
    FILE * bin = std::fopen((out_path + ".bin").c_str(), "ab");
    FILE * idx = std::fopen((out_path + ".idx.jsonl").c_str(), "ab");
    if (!in || !bin || !idx) {
        LOG_ERR("cannot open input/output files\n");
        return 1;
    }
    std::vector<ggml_fp16_t> row(n_embd);
    std::vector<ggml_bf16_t> frow(dflash ? n_feat_row : 0);
    std::vector<ggml_fp16_t> hbuf;   // DFlash mode: the sequence's h rows, written after its feature rows
    std::vector<float> feat_absmax(layers.size(), 0.0f);
    long long n_nonfinite = 0, n_h_sat = 0;
    std::string line;
    long long total_tokens = 0, total_keep = 0, total_h = 0;
    int n_done = 0;
    const int64_t t_start = ggml_time_us();
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const json rec = json::parse(line);
        const std::string id   = rec.at("id").get<std::string>();
        const std::string text = rec.at("text").get<std::string>();

        std::vector<llama_token> tokens = common_tokenize(ctx, text, /*add_special*/ false, /*parse_special*/ true);
        const int n_full = (int) tokens.size();
        const int n = std::min<int>(n_full, n_ctx);
        tokens.resize(n);

        // byte offset of every token, from the detokenized pieces (byte-level BPE reproduces the text exactly)
        std::vector<uint8_t> train(n, 0), keep(n, 0), keep_h(n, 0);
        size_t off = 0;
        std::string rebuilt;
        for (int i = 0; i < n; i++) {
            const size_t start = off;
            const std::string piece = common_token_to_piece(ctx, tokens[i], true);
            rebuilt += piece;
            off += piece.size();
            for (const auto & span : rec.at("train")) {
                if (start >= span[0].get<size_t>() && start < span[1].get<size_t>()) {
                    train[i] = 1;
                    break;
                }
            }
        }
        if (!dflash) {
            // the head at position t sees h[t] and predicts token t+2 (step 1) .. t+1+k (step k): keep h for the
            // positions up to `ahead` before each trainable token
            for (int i = 0; i < n; i++) {
                if (train[i]) {
                    for (int j = std::max(0, i - 1 - ahead); j < i; j++) {
                        keep[j] = 1;
                    }
                    keep[i] = 1;
                }
            }
        } else {
            // keep_f (in `keep`): feature row p is visible to an anchor a (train[a]) iff a - window <= p < a
            int next_train = -1;   // smallest train position > p, scanning backwards
            for (int p = n - 1; p >= 0; p--) {
                if (next_train >= 0 && next_train - p <= window) {
                    keep[p] = 1;
                }
                if (train[p]) {
                    next_train = p;
                }
            }
            for (int j = 0; j + 1 < n; j++) {
                keep_h[j] = train[j] && train[j + 1];
            }
        }
        // the pieces must rebuild the text byte for byte, otherwise the train mask would be shifted
        const bool bytes_ok = (n < n_full ? off <= text.size() : off == text.size()) &&
                              text.compare(0, rebuilt.size(), rebuilt) == 0;
        if (!bytes_ok) {
            LOG_WRN("%s: detokenized %zu bytes, text has %zu: train mask may be misaligned, skipped\n",
                    id.c_str(), off, text.size());
            continue;
        }
        int n_keep = 0, n_h = 0;
        for (int i = 0; i < n; i++) {
            n_keep += keep[i];
            n_h += keep_h[i];
        }
        if (n_keep == 0 || (dflash && n_h == 0)) {
            continue;
        }

        // 64-bit offset: ftell() returns a 32-bit long on Windows, which wraps past 2 GB
#ifdef _WIN32
        const int64_t pos_bin = _ftelli64(bin);
#else
        const int64_t pos_bin = ftello(bin);
#endif
        if (dflash) {
            const int32_t hdr[6] = { 0x32464644, n, n_keep, n_h, (int32_t) layers.size(), n_embd };
            std::fwrite(hdr, sizeof(int32_t), 6, bin);
            std::fwrite(tokens.data(), sizeof(llama_token), n, bin);
            std::fwrite(train.data(), 1, n, bin);
            std::fwrite(keep.data(), 1, n, bin);
            std::fwrite(keep_h.data(), 1, n, bin);
            hbuf.clear();
            hbuf.reserve((size_t) n_h * n_embd);
        } else {
            std::fwrite(&n, sizeof(int32_t), 1, bin);
            std::fwrite(&n_keep, sizeof(int32_t), 1, bin);
            std::fwrite(tokens.data(), sizeof(llama_token), n, bin);
            std::fwrite(train.data(), 1, n, bin);
            std::fwrite(keep.data(), 1, n, bin);
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        common_batch batch(ctx);
        for (int b0 = 0; b0 < n; b0 += n_batch) {
            const int b1 = std::min(n, b0 + n_batch);
            batch.clear();
            for (int i = b0; i < b1; i++) {
                // DFlash mode needs result_norm only at keep_h; the last token of every batch is an output as well
                // so that no ubatch runs with zero outputs (its layer inputs are still extracted)
                const bool out = dflash ? (keep_h[i] != 0 || i == b1 - 1) : keep[i] != 0;
                batch.add(tokens[i], i, 0, out);
            }
            const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            // DFlash mode: any non-zero return (1 = no KV slot, 2 = aborted) would leave the previous batch's rows
            // in the output buffers, so it is fatal too; MTP mode keeps its original check
            if (rc < 0 || (dflash && rc != 0)) {
                LOG_ERR("decode failed for %s (rc %d)\n", id.c_str(), rc);
                return 1;
            }
            if (!dflash) {
                for (int i = b0; i < b1; i++) {
                    if (!keep[i]) {
                        continue;
                    }
                    const float * e = llama_get_embeddings_ith(ctx, i - b0);
                    GGML_ASSERT(e != nullptr && "no embeddings: pooling must be none");
                    ggml_fp32_to_fp16_row(e, row.data(), n_embd);
                    std::fwrite(row.data(), sizeof(ggml_fp16_t), n_embd, bin);
                }
                continue;
            }
            // feature rows: rows of the layer-input buffers are indexed by the token's place in this batch
            std::vector<const float *> lin(layers.size());
            for (size_t k = 0; k < layers.size(); k++) {
                lin[k] = llama_get_embeddings_layer_inp(ctx, (uint32_t) layers[k]);
                GGML_ASSERT(lin[k] != nullptr && "layer input not extracted");
            }
            for (int i = b0; i < b1; i++) {
                if (keep[i]) {
                    for (size_t k = 0; k < layers.size(); k++) {
                        const float * src = lin[k] + (size_t) (i - b0) * n_embd;
                        for (int c = 0; c < n_embd; c++) {
                            const float v = src[c];
                            if (!std::isfinite(v)) {
                                n_nonfinite++;
                            } else if (std::fabs(v) > feat_absmax[k]) {
                                feat_absmax[k] = std::fabs(v);
                            }
                        }
                        ggml_fp32_to_bf16_row(src, frow.data() + k * n_embd, n_embd);
                    }
                    std::fwrite(frow.data(), sizeof(ggml_bf16_t), n_feat_row, bin);
                }
                if (keep_h[i]) {
                    const float * e = llama_get_embeddings_ith(ctx, i - b0);
                    GGML_ASSERT(e != nullptr && "no embeddings: pooling must be none");
                    for (int c = 0; c < n_embd; c++) {
                        n_h_sat += std::fabs(e[c]) > 65504.0f;
                    }
                    ggml_fp32_to_fp16_row(e, row.data(), n_embd);
                    hbuf.insert(hbuf.end(), row.begin(), row.end());
                }
            }
        }
        if (dflash) {
            GGML_ASSERT(hbuf.size() == (size_t) n_h * n_embd);
            std::fwrite(hbuf.data(), sizeof(ggml_fp16_t), hbuf.size(), bin);
        }
        std::fflush(bin);
        json ix = {{"id", id}, {"offset", pos_bin}, {"n_tokens", n}, {"n_keep", n_keep},
                   {"truncated", n < n_full}};
        if (dflash) {
            ix["n_h"] = n_h;
            ix["format"] = "dflash2";
            ix["layers"] = layers;
            ix["window"] = window;
        }
        std::fprintf(idx, "%s\n", ix.dump().c_str());
        std::fflush(idx);
        total_tokens += n;
        total_keep += n_keep;
        total_h += n_h;
        n_done++;
        if (n_done % 20 == 0) {
            const double s = (ggml_time_us() - t_start) / 1e6;
            LOG_INF("%d sequences, %lld tokens (%.0f tok/s), %lld kept, %lld h rows\n", n_done, total_tokens,
                    total_tokens / s, total_keep, total_h);
        }
    }
    std::fclose(bin);
    std::fclose(idx);
    LOG_INF("done: %d sequences, %lld tokens, %lld kept rows, %lld h rows (n_embd %d)\n", n_done, total_tokens,
            total_keep, total_h, n_embd);
    if (dflash) {
        for (size_t k = 0; k < layers.size(); k++) {
            LOG_INF("layer %d input: max |value| %.1f\n", layers[k], feat_absmax[k]);
        }
        LOG_INF("non-finite feature values: %lld, h values beyond fp16 range: %lld\n", n_nonfinite, n_h_sat);
    }
    llama_backend_free();
    return 0;
}
