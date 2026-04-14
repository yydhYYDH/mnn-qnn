#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include "MNN_generated.h"

using namespace MNN;
using namespace MNN::Express;

namespace {

struct KVMeta {
    enum {
        NoChange,
        PendingWrite,
        PendingRead
    };

    size_t block = 4096;
    size_t previous = 0;
    size_t remove = 0;
    int * reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    std::string file_name;
    int file_flag = NoChange;
    int seqlen_in_disk = 0;
    int layer_index = 0;
    int layer_nums = 0;
    std::vector<int> reserveHost;

    void sync() {
        int revert_number = 0;
        for (int i = 0; i < n_reserve; ++i) {
            revert_number += reserve[2 * i + 1];
        }
        previous = previous - remove + add + revert_number;
        n_reserve = 0;
        reserve = nullptr;
        remove = 0;
        add = 0;
    }
};

struct Options {
    int seq_len = 32;
    int kv_len = 32;
    int num_heads = 32;
    int head_dim = 128;
    int attention_option = 8;
    int num_threads = 1;
    int causal = 1;
    std::string dump_dir;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--seq-len N] [--kv-len N] [--num-heads N] [--head-dim N]\n"
                 "          [--attention-option N] [--num-threads N] [--causal 0|1]\n"
                 "          [--dump-dir DIR]\n",
                 argv0);
}

static bool parse_int_arg(const char * key, const char * value, int * output) {
    if (value == nullptr) {
        std::fprintf(stderr, "missing value for %s\n", key);
        return false;
    }
    *output = std::atoi(value);
    return true;
}

static bool parse_args(int argc, char ** argv, Options * opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--seq-len") {
            const char * value = i + 1 < argc ? argv[i + 1] : nullptr;
            if (!parse_int_arg(argv[i], value, &opt->seq_len)) {
                return false;
            }
            ++i;
        } else if (arg == "--kv-len") {
            const char * value = i + 1 < argc ? argv[i + 1] : nullptr;
            if (!parse_int_arg(argv[i], value, &opt->kv_len)) {
                return false;
            }
            ++i;
        } else if (arg == "--num-heads") {
            const char * value = i + 1 < argc ? argv[i + 1] : nullptr;
            if (!parse_int_arg(argv[i], value, &opt->num_heads)) {
                return false;
            }
            ++i;
        } else if (arg == "--head-dim") {
            const char * value = i + 1 < argc ? argv[i + 1] : nullptr;
            if (!parse_int_arg(argv[i], value, &opt->head_dim)) {
                return false;
            }
            ++i;
        } else if (arg == "--attention-option") {
            const char * value = i + 1 < argc ? argv[i + 1] : nullptr;
            if (!parse_int_arg(argv[i], value, &opt->attention_option)) {
                return false;
            }
            ++i;
        } else if (arg == "--num-threads") {
            const char * value = i + 1 < argc ? argv[i + 1] : nullptr;
            if (!parse_int_arg(argv[i], value, &opt->num_threads)) {
                return false;
            }
            ++i;
        } else if (arg == "--causal") {
            const char * value = i + 1 < argc ? argv[i + 1] : nullptr;
            if (!parse_int_arg(argv[i], value, &opt->causal)) {
                return false;
            }
            ++i;
        } else if (arg == "--dump-dir") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for --dump-dir\n");
                return false;
            }
            opt->dump_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            return false;
        }
    }

    if (opt->seq_len <= 0 || opt->kv_len <= 0 || opt->num_heads <= 0 || opt->head_dim <= 0) {
        std::fprintf(stderr, "all dimensions must be positive\n");
        return false;
    }
    if (opt->kv_len < opt->seq_len) {
        std::fprintf(stderr, "kv_len must be >= seq_len\n");
        return false;
    }
    return true;
}

static float deterministic_value(size_t index, int salt) {
    const uint64_t x = static_cast<uint64_t>(index);
    const int term0 = static_cast<int>((x * 17 + static_cast<uint64_t>(salt) * 13) % 127) - 63;
    const int term1 = static_cast<int>((x * 11 + static_cast<uint64_t>(salt) * 7) % 31) - 15;
    return term0 * (1.0f / 32.0f) + term1 * (1.0f / 1024.0f);
}

static float mask_value(int row, int col, const Options & opt) {
    if (!opt.causal) {
        return 0.0f;
    }
    const int visible_prefix = opt.kv_len - opt.seq_len;
    return (col - row <= visible_prefix) ? 0.0f : -1.0e9f;
}

static bool mkdirs(const std::string & path) {
    if (path.empty()) {
        return true;
    }
    size_t cursor = 0;
    while (cursor < path.size()) {
        cursor = path.find('/', cursor + 1);
        const std::string current = cursor == std::string::npos ? path : path.substr(0, cursor);
        if (current.empty()) {
            continue;
        }
        if (::mkdir(current.c_str(), 0777) != 0 && errno != EEXIST) {
            std::fprintf(stderr, "mkdir failed for %s errno=%d\n", current.c_str(), errno);
            return false;
        }
        if (cursor == std::string::npos) {
            break;
        }
    }
    return true;
}

static bool write_tensor(const std::string & dir, const std::string & name,
                         const std::vector<float> & data, const std::vector<int> & shape) {
    if (dir.empty()) {
        return true;
    }
    if (!mkdirs(dir)) {
        return false;
    }
    const std::string bin_path = dir + "/" + name + ".bin";
    const std::string meta_path = dir + "/" + name + ".meta.txt";
    FILE * fp = std::fopen(bin_path.c_str(), "wb");
    if (fp == nullptr) {
        std::fprintf(stderr, "failed to open %s\n", bin_path.c_str());
        return false;
    }
    if (!data.empty()) {
        std::fwrite(data.data(), sizeof(float), data.size(), fp);
    }
    std::fclose(fp);

    fp = std::fopen(meta_path.c_str(), "w");
    if (fp == nullptr) {
        std::fprintf(stderr, "failed to open %s\n", meta_path.c_str());
        return false;
    }
    std::fprintf(fp, "name=%s\n", name.c_str());
    std::fprintf(fp, "dtype=f32\n");
    std::fprintf(fp, "rank=%zu\n", shape.size());
    std::fprintf(fp, "shape=");
    for (size_t i = 0; i < shape.size(); ++i) {
        std::fprintf(fp, "%s%d", i == 0 ? "" : ",", shape[i]);
    }
    std::fprintf(fp, "\nbytes=%zu\n", data.size() * sizeof(float));
    std::fclose(fp);
    return true;
}

static VARP make_input(const std::vector<float> & data, const std::vector<int> & shape, halide_type_t type = halide_type_of<float>()) {
    VARP var = _Input(shape, NCHW, type);
    float * ptr = var->writeMap<float>();
    std::memcpy(ptr, data.data(), data.size() * sizeof(float));
    var->unMap();
    return var;
}

static std::shared_ptr<Module> make_attention_module(const Options & opt, KVMeta * meta) {
    auto q = _Input();
    auto k = _Input();
    auto v = _Input();
    auto mask = _Input();

    std::shared_ptr<OpT> attention(new OpT);
    attention->type = OpType_Attention;
    attention->main.type = OpParameter_AttentionParam;
    attention->main.value = new AttentionParamT;
    attention->main.AsAttentionParam()->kv_cache = true;

    auto out = Variable::create(Expr::create(attention.get(), {q, k, v, mask}));
    auto buffer = Variable::save({out});

    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = opt.num_threads;
    BackendConfig backend_config;
    backend_config.precision = BackendConfig::Precision_Normal;
    backend_config.memory = BackendConfig::Memory_Normal;
    backend_config.power = BackendConfig::Power_Normal;
    config.backendConfig = &backend_config;

    std::shared_ptr<Executor::RuntimeManager> runtime(Executor::RuntimeManager::createRuntimeManager(config));
    runtime->setHintPtr(Interpreter::KVCACHE_INFO, meta);
    runtime->setHint(Interpreter::ATTENTION_OPTION, opt.attention_option);

    return std::shared_ptr<Module>(Module::load({}, {}, reinterpret_cast<uint8_t *>(buffer.data()), buffer.size(), runtime));
}

}  // namespace

int main(int argc, char ** argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        usage(argv[0]);
        return 1;
    }

    const int q_size = opt.seq_len * opt.num_heads * opt.head_dim;
    const int kv_size = opt.kv_len * opt.num_heads * opt.head_dim;
    const int mask_size = opt.seq_len * opt.kv_len;

    std::vector<float> q_data(q_size);
    std::vector<float> k_data(kv_size);
    std::vector<float> v_data(kv_size);
    std::vector<float> mask_data(mask_size);

    for (int i = 0; i < q_size; ++i) {
        q_data[i] = deterministic_value(static_cast<size_t>(i), 1);
    }
    for (int i = 0; i < kv_size; ++i) {
        k_data[i] = deterministic_value(static_cast<size_t>(i), 2);
        v_data[i] = deterministic_value(static_cast<size_t>(i), 3);
    }
    for (int row = 0; row < opt.seq_len; ++row) {
        for (int col = 0; col < opt.kv_len; ++col) {
            mask_data[row * opt.kv_len + col] = mask_value(row, col, opt);
        }
    }

    KVMeta meta;
    meta.previous = static_cast<size_t>(opt.kv_len - opt.seq_len);
    meta.add = static_cast<size_t>(opt.seq_len);

    auto module = make_attention_module(opt, &meta);
    auto q = make_input(q_data, {1, opt.seq_len, opt.num_heads, opt.head_dim});
    auto k = make_input(k_data, {1, opt.kv_len, opt.num_heads, opt.head_dim});
    auto v = make_input(v_data, {1, opt.kv_len, opt.num_heads, opt.head_dim});
    VARP mask;
    if (opt.causal) {
        std::vector<float> causal_mask_scalar(1, 0.0f);
        mask = make_input(causal_mask_scalar, {});
    } else {
        mask = make_input(mask_data, {1, 1, opt.seq_len, opt.kv_len});
    }

    MNN::Timer timer;
    auto outputs = module->onForward({q, k, v, mask});
    meta.sync();
    const float elapsed_ms = timer.durationInUs() / 1000.0f;
    if (outputs.empty()) {
        std::fprintf(stderr, "attention op returned no outputs\n");
        return 2;
    }

    VARP result = outputs[0];
    auto info = result->getInfo();
    if (info != nullptr && info->order == NC4HW4 && info->dim.size() > 1) {
        result = _Convert(result, NCHW);
    }
    info = result->getInfo();
    if (info != nullptr && info->type.code != halide_type_float) {
        result = _Cast<float>(result);
    }
    result.fix(VARP::CONSTANT);

    const float * out_ptr = result->readMap<float>();
    if (out_ptr == nullptr) {
        std::fprintf(stderr, "failed to map output tensor\n");
        return 3;
    }

    std::vector<float> out_data(q_size);
    std::memcpy(out_data.data(), out_ptr, out_data.size() * sizeof(float));
    outputs[0]->unMap();

    if (!opt.dump_dir.empty()) {
        if (!write_tensor(opt.dump_dir, "attn-query", q_data, {1, opt.seq_len, opt.num_heads, opt.head_dim}) ||
            !write_tensor(opt.dump_dir, "attn-key", k_data, {1, opt.kv_len, opt.num_heads, opt.head_dim}) ||
            !write_tensor(opt.dump_dir, "attn-value", v_data, {1, opt.kv_len, opt.num_heads, opt.head_dim}) ||
            !write_tensor(opt.dump_dir, "attn-mask", mask_data, {1, 1, opt.seq_len, opt.kv_len}) ||
            !write_tensor(opt.dump_dir, "attn-output", out_data, {1, opt.seq_len, opt.num_heads, opt.head_dim})) {
            return 4;
        }
    }

    std::printf("MNN attention unit test finished\n");
    std::printf("seq_len=%d kv_len=%d num_heads=%d head_dim=%d causal=%d attention_option=%d num_threads=%d kv_previous=%zu kv_add=%zu mask_mode=%s\n",
                opt.seq_len, opt.kv_len, opt.num_heads, opt.head_dim, opt.causal, opt.attention_option,
                opt.num_threads, meta.previous, meta.add, opt.causal ? "scalar-causal" : "dense");
    std::printf("elapsed_ms=%.4f\n", elapsed_ms);
    if (!out_data.empty()) {
        std::printf("output_first=%.9g output_last=%.9g\n", out_data.front(), out_data.back());
    }
    if (info != nullptr) {
        std::printf("output_order=%d output_rank=%zu\n", info->order, info->dim.size());
    }
    if (!opt.dump_dir.empty()) {
        std::printf("dump_dir=%s\n", opt.dump_dir.c_str());
    }
    return 0;
}
