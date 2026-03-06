#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <zstd.h>
#include <nlohmann/json.hpp>
#include <gflags/gflags.h>

using json = nlohmann::json;

DEFINE_int32(limit, 0, "Maximum number of positions to convert; 0 for no limit");

/*
# https://database.lichess.org/#evals
wget https://database.lichess.org/lichess_db_eval.jsonl.zst

g++ -std=c++20 -O3 -march=native src/ConvertLichessDbMain.cpp -o convert_lichess -L/usr/bin -I/usr/include -I/usr/local/include -lzstd -lgflags

./convert_lichess path/to/lichess_db_eval.jsonl.zst output.txt
*/

bool process_line(const std::string& line, std::ofstream& out) {
    if (line.empty()) return false;
    try {
        auto j = json::parse(line);
        if (!j.contains("fen") || !j.contains("evals")) return false;
        
        std::string fen = j["fen"];
        auto evals = j["evals"];
        if (evals.empty()) return false;
        
        if (evals[0].contains("depth")) {
            int depth = evals[0]["depth"];
            if (depth < 12) return false;
        }
        
        auto pvs = evals[0]["pvs"];
        if (pvs.empty()) return false;
        
        auto pv = pvs[0];
        float score = 0.0f;
        
        if (pv.contains("mate")) {
            int mate = pv["mate"];
            // If mate > 0, white is mating. If mate < 0, black is mating.
            score = (mate > 0) ? 100.0f : -100.0f;
        } else if (pv.contains("cp")) {
            int cp = pv["cp"];
            score = cp / 100.0f;
        } else {
            return false;
        }
        
        char buf[64];
        snprintf(buf, sizeof(buf), "%+.2f", score);
        out << fen << "|" << buf << "\n";
        return true;
    } catch (const std::exception& e) {
        // ignore parsing errors and continue
        return false;
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.jsonl.zst> <output.txt> [--limit=<num>]\n";
        return 1;
    }
    
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input file: " << argv[1] << "\n";
        return 1;
    }
    
    std::ofstream out(argv[2]);
    if (!out) {
        std::cerr << "Failed to open output file: " << argv[2] << "\n";
        return 1;
    }
    
    size_t const buffInSize = ZSTD_DStreamInSize();
    std::vector<char> buffIn(buffInSize);
    
    size_t const buffOutSize = ZSTD_DStreamOutSize();
    std::vector<char> buffOut(buffOutSize);
    
    ZSTD_DCtx* const dctx = ZSTD_createDCtx();
    if (!dctx) {
        std::cerr << "Failed to create ZSTD context\n";
        return 1;
    }
    
    std::string uncompressed_buffer;
    size_t toRead = buffInSize;
    size_t readCount;
    int num_processed = 0;
    
    while (in) {
        in.read(buffIn.data(), buffInSize);
        readCount = in.gcount();
        if (readCount == 0) break;
        
        ZSTD_inBuffer input = { buffIn.data(), readCount, 0 };
        while (input.pos < input.size) {
            ZSTD_outBuffer output = { buffOut.data(), buffOutSize, 0 };
            size_t const ret = ZSTD_decompressStream(dctx, &output , &input);
            if (ZSTD_isError(ret)) {
                std::cerr << "ZSTD decompression error: " << ZSTD_getErrorName(ret) << "\n";
                ZSTD_freeDCtx(dctx);
                return 1;
            }
            
            // output.pos contains the number of bytes decompressed this time
            for (size_t i = 0; i < output.pos; ++i) {
                char c = ((char*)output.dst)[i];
                if (c == '\n') {
                    if (process_line(uncompressed_buffer, out)) {
                        num_processed++;
                        if (num_processed % 100000 == 0) {
                            std::cout << "Processed " << num_processed << " positions...\n";
                        }
                        if (FLAGS_limit > 0 && num_processed >= FLAGS_limit) {
                            goto finish;
                        }
                    }
                    uncompressed_buffer.clear();
                } else {
                    uncompressed_buffer += c;
                }
            }
        }
    }
    
    if (!uncompressed_buffer.empty()) {
        if (process_line(uncompressed_buffer, out)) {
            num_processed++;
        }
    }
    
finish:
    ZSTD_freeDCtx(dctx);
    std::cout << "Successfully converted " << num_processed << " positions from " << argv[1] << " to " << argv[2] << "\n";
    return 0;
}
