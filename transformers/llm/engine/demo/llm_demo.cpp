//
//  llm_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sstream>
#include <stdlib.h>
#include <initializer_list>
//#define LLM_SUPPORT_AUDIO
#ifdef LLM_SUPPORT_AUDIO
#include "audio/audio.hpp"
#endif
using namespace MNN::Transformer;

static bool mkdirs_if_needed(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    size_t cursor = 0;
    while (cursor < path.size()) {
        cursor = path.find('/', cursor + 1);
        std::string current = cursor == std::string::npos ? path : path.substr(0, cursor);
        if (current.empty()) {
            continue;
        }
        if (mkdir(current.c_str(), 0777) != 0 && errno != EEXIST) {
            MNN_ERROR("mkdir failed for %s errno=%d\n", current.c_str(), errno);
            return false;
        }
        if (cursor == std::string::npos) {
            break;
        }
    }
    return true;
}

static std::string dump_dir_from_env() {
    const char* value = getenv("MNN_LLM_DUMP_PROMPT_DIR");
    if (value == nullptr || value[0] == '\0') {
        return "";
    }
    return value;
}

static bool dump_only_from_env() {
    const char* value = getenv("MNN_LLM_DUMP_ONLY");
    return value != nullptr && value[0] != '\0' &&
           strcmp(value, "0") != 0 &&
           strcmp(value, "false") != 0 &&
           strcmp(value, "False") != 0 &&
           strcmp(value, "FALSE") != 0;
}

static void dump_prompt_inputs(Llm* llm, const std::string& prompt, size_t prompt_index) {
    const std::string dump_dir = dump_dir_from_env();
    if (dump_dir.empty()) {
        return;
    }
    if (!mkdirs_if_needed(dump_dir)) {
        return;
    }

    char prefix[64];
    snprintf(prefix, sizeof(prefix), "prompt-%04zu", prompt_index);

    std::ofstream prompt_os(dump_dir + "/" + std::string(prefix) + ".prompt.txt");
    prompt_os << prompt;
    prompt_os.close();

    std::vector<int> input_ids = llm->tokenizer_encode(prompt);
    std::ofstream ids_os(dump_dir + "/" + std::string(prefix) + ".input_ids.txt");
    for (size_t i = 0; i < input_ids.size(); ++i) {
        ids_os << input_ids[i] << "\n";
    }
    ids_os.close();

    auto embeds = llm->embedding(input_ids);
    auto info = embeds->getInfo();
    auto* data = embeds->readMap<float>();
    if (info == nullptr || data == nullptr) {
        MNN_ERROR("Failed to map embedding for prompt dump\n");
        return;
    }

    const std::string stem = dump_dir + "/" + std::string(prefix) + ".embedding";
    std::ofstream raw_os(stem + ".bin", std::ios::binary);
    raw_os.write(reinterpret_cast<const char*>(data), info->size * sizeof(float));
    raw_os.close();

    std::ofstream meta_os(stem + ".meta.txt");
    meta_os << "dtype=float32\n";
    meta_os << "size=" << info->size << "\n";
    meta_os << "dim=";
    for (size_t i = 0; i < info->dim.size(); ++i) {
        meta_os << (i == 0 ? "" : ",") << info->dim[i];
    }
    meta_os << "\n";
    meta_os.close();

    MNN_PRINT("[dump] wrote prompt/input_ids/embedding for prompt %zu to %s\n", prompt_index, dump_dir.c_str());
}

static void tuning_prepare(Llm* llm) {
    MNN_PRINT("Prepare for tuning opt Begin\n");
    llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    MNN_PRINT("Prepare for tuning opt End\n");
}

std::vector<std::vector<std::string>> parse_csv(const std::vector<std::string>& lines) {
    std::vector<std::vector<std::string>> csv_data;
    std::string line;
    std::vector<std::string> row;
    std::string cell;
    bool insideQuotes = false;
    bool startCollecting = false;

    // content to stream
    std::string content = "";
    for (auto line : lines) {
        content = content + line + "\n";
    }
    std::istringstream stream(content);

    while (stream.peek() != EOF) {
        char c = stream.get();
        if (c == '"') {
            if (insideQuotes && stream.peek() == '"') { // quote
                cell += '"';
                stream.get(); // skip quote
            } else {
                insideQuotes = !insideQuotes; // start or end text in quote
            }
            startCollecting = true;
        } else if (c == ',' && !insideQuotes) { // end element, start new element
            row.push_back(cell);
            cell.clear();
            startCollecting = false;
        } else if ((c == '\n' || stream.peek() == EOF) && !insideQuotes) { // end line
            row.push_back(cell);
            csv_data.push_back(row);
            cell.clear();
            row.clear();
            startCollecting = false;
        } else {
            cell += c;
            startCollecting = true;
        }
    }
    return csv_data;
}

static int benchmark(Llm* llm, const std::vector<std::string>& prompts, int max_token_number) {
    int prompt_len = 0;
    int decode_len = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    int64_t sample_time = 0;
    // llm->warmup();
    auto context = llm->getContext();
    if (max_token_number > 0) {
        llm->set_config("{\"max_new_tokens\":1}");
    }
#ifdef LLM_SUPPORT_AUDIO
    std::vector<float> waveform;
    llm->setWavformCallback([&](const float* ptr, size_t size, bool last_chunk) {
        waveform.reserve(waveform.size() + size);
        waveform.insert(waveform.end(), ptr, ptr + size);
        if (last_chunk) {
            auto waveform_var = MNN::Express::_Const(waveform.data(), {(int)waveform.size()}, MNN::Express::NCHW, halide_type_of<float>());
            MNN::AUDIO::save("output.wav", waveform_var, 24000);
            waveform.clear();
        }
        return true;
    });
#endif
    for (int i = 0; i < prompts.size(); i++) {
        auto prompt = prompts[i];
     // #define MIMO_NO_THINKING
     #ifdef MIMO_NO_THINKING
        // update config.json and llm_config.json if need. example:
        llm->set_config("{\"assistant_prompt_template\":\"<|im_start|>assistant\\n<think>\\n</think>\%s<|im_end|>\\n\"}");
        prompt = prompt + "<think>\n</think>";
     #endif

        // prompt start with '#' will be ignored
        if (prompt.substr(0, 1) == "#") {
            continue;
        }

        dump_prompt_inputs(llm, prompt, static_cast<size_t>(i));
        if (dump_only_from_env()) {
            continue;
        }
        
        if (max_token_number >= 0) {
            llm->response(prompt, &std::cout, nullptr, 0);
            while (!llm->stoped() && context->gen_seq_len < max_token_number) {
                llm->generate(1);
            }
        } else {
            llm->response(prompt);
        }
        prompt_len += context->prompt_len;
        decode_len += context->gen_seq_len;
        prefill_time += context->prefill_us;
        decode_time += context->decode_us;
        sample_time += context->sample_us;
    }
    llm->generateWavform();

    float vision_s = context->vision_us / 1e6;
    float audio_s = context->audio_us / 1e6;
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    float sample_s = sample_time / 1e6;
    float vision_speed = 0.0f;
    if (context->pixels_mp > 0.0f) {
        vision_speed = context->pixels_mp / vision_s;
    }
    float audio_speed = 0.0f;
    if (context->audio_input_s > 0.0f) {
        audio_speed = context->audio_input_s / audio_s;
    }
    MNN_PRINT("\n#################################\n");
    MNN_PRINT("prompt tokens num = %d\n", prompt_len);
    MNN_PRINT("decode tokens num = %d\n", decode_len);
    MNN_PRINT(" vision time = %.2f s\n", vision_s);
    MNN_PRINT(" pixels_mp = %.2f MP\n", context->pixels_mp);
    MNN_PRINT("  audio process time = %.2f s\n", audio_s);
    MNN_PRINT("  audio input time = %.2f s\n", context->audio_input_s);
    MNN_PRINT("prefill time = %.2f s\n", prefill_s);
    MNN_PRINT(" decode time = %.2f s\n", decode_s);
    MNN_PRINT(" sample time = %.2f s\n", sample_s);
    MNN_PRINT("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    MNN_PRINT(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    MNN_PRINT(" vision speed = %.3f MP/s\n", vision_speed);
    MNN_PRINT(" audio RTF = %.3f \n", audio_s / context->audio_input_s);
    MNN_PRINT("##################################\n");
    return 0;
}

static int ceval(Llm* llm, const std::vector<std::string>& lines, std::string filename) {
    auto csv_data = parse_csv(lines);
    int right = 0, wrong = 0;
    std::vector<std::string> answers;
    for (int i = 1; i < csv_data.size(); i++) {
        const auto& elements = csv_data[i];
        std::string prompt = elements[1];
        prompt += "\n\nA. " + elements[2];
        prompt += "\nB. " + elements[3];
        prompt += "\nC. " + elements[4];
        prompt += "\nD. " + elements[5];
        prompt += "\n\n";
        MNN_PRINT("%s", prompt.c_str());
        MNN_PRINT("## 进度: %d / %lu\n", i, lines.size() - 1);
        std::ostringstream lineOs;
        llm->response(prompt.c_str(), &lineOs);
        auto line = lineOs.str();
        MNN_PRINT("%s", line.c_str());
        answers.push_back(line);
    }
    {
        auto position = filename.rfind("/");
        if (position != std::string::npos) {
            filename = filename.substr(position + 1, -1);
        }
        position = filename.find("_val");
        if (position != std::string::npos) {
            filename.replace(position, 4, "_res");
        }
        std::cout << "store to " << filename << std::endl;
    }
    std::ofstream ofp(filename);
    ofp << "id,answer" << std::endl;
    for (int i = 0; i < answers.size(); i++) {
        auto& answer = answers[i];
        ofp << i << ",\""<< answer << "\"" << std::endl;
    }
    ofp.close();
    return 0;
}

static int eval(Llm* llm, std::string prompt_file, int max_token_number) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
//#define LLM_DEMO_ONELINE
#ifdef LLM_DEMO_ONELINE
    std::ostringstream tempOs;
    tempOs << prompt_fs.rdbuf();
    prompt = tempOs.str();
    prompts = {prompt};
#else
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.empty()) {
            continue;
        }
        if (prompt.back() == '\r') {
            prompt.pop_back();
        }
        prompts.push_back(prompt);
    }
#endif
    prompt_fs.close();
    if (prompts.empty()) {
        return 1;
    }
    // ceval
    if (prompts[0] == "id,question,A,B,C,D,answer") {
        return ceval(llm, prompts, prompt_file);
    }
    return benchmark(llm, prompts, max_token_number);
}

void chat(Llm* llm) {
    ChatMessages messages;
    messages.emplace_back("system", "You are a helpful assistant.");
    auto context = llm->getContext();
    while (true) {
        std::cout << "\nUser: ";
        std::string user_str;
        std::getline(std::cin, user_str);
        if (user_str == "/exit") {
            return;
        }
        if (user_str == "/reset") {
            llm->reset();
            std::cout << "\nA: reset done." << std::endl;
            continue;
        }
        messages.emplace_back("user", user_str);
        std::cout << "\nA: " << std::flush;
        llm->response(messages);
        auto assistant_str = context->generate_str;
        messages.emplace_back("assistant", assistant_str);
    }
}
int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json <prompt.txt>" << std::endl;
        return 0;
    }

    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    {
        AUTOTIME;
        bool res = llm->load();
        if (!res) {
            MNN_ERROR("LLM init error\n");
            return 0;
        }
    }
    if (true) {
        AUTOTIME;
        tuning_prepare(llm.get());
    }
    if (argc < 3) {
        chat(llm.get());
        return 0;
    }
    int max_token_number = -1;
    if (argc >= 4) {
        std::istringstream os(argv[3]);
        os >> max_token_number;
    }
    if (argc >= 5) {
        MNN_PRINT("Set not thinking, only valid for Qwen3\n");
        llm->set_config(R"({
            "jinja": {
                "context": {
                    "enable_thinking":false
                }
            }
        })");
    }
    std::string prompt_file = argv[2];
    llm->set_config(R"({
        "async":false
    })");
    return eval(llm.get(), prompt_file, max_token_number);
}
