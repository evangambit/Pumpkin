#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <zstd.h>
#include <nlohmann/json.hpp>
#include <gflags/gflags.h>

#include "game/movegen/movegen.h"
#include "StringUtils.h"

using json = nlohmann::json;

typedef ChessEngine::Position Position;
typedef ChessEngine::Move Move;

DEFINE_int32(limit, 0, "Maximum number of positions to convert; 0 for no limit");

/*
# https://database.lichess.org/#evals
wget https://database.lichess.org/lichess_db_eval.jsonl.zst

sh build.sh convert_lichess src/ConvertLichessDbMain.cpp -lzstd

./convert_lichess path/to/lichess_db_eval.jsonl.zst output.txt
*/



bool process_line(const std::string& line, std::ofstream& out) {
    if (line.empty()) return false;
    try {
        auto j = json::parse(line);
        if (!j.contains("fen") || !j.contains("evals")) return false;
        
        std::string fen = j["fen"];
        auto evals = j["evals"];
        if (evals.empty()) {
            return false;
        }

        const size_t num_evals = evals.size();
        size_t index = rand() % num_evals;
        auto eval = evals[index];

        int retry;
        for (retry = 0; retry < 5; ++retry) {
            int depth = eval["depth"];
            int num_pvs = eval["pvs"].size();
            // We prefer num_pvs > 1 for increased diversity.
            if (depth >= 12 && num_pvs > 1) {
                break;
            }
            index = rand() % num_evals;
            eval = evals[index];
        }
        
        if (retry == 5) {
            // Failed to find a good eval after 5 tries, skip this position.
            return false;
        }
        
        auto pvs = eval["pvs"];        
        auto pv = pvs[rand() % pvs.size()];
        float score = 0.0f;
        
        if (pv.contains("mate")) {
            return false;
        } else if (pv.contains("cp")) {
            int cp = pv["cp"];
            score = cp / 100.0f;
        } else {
            return false;
        }

        Position pos(fen);
        std::vector<std::string> line = split(pv["line"], ' ');
        if (line.size() < 2) {
            // We want at least 2 moves in the PV for our training data.
            return false;
        }
        Move firstMove = ChessEngine::uci_to_move(pos, line[0]);
        if (firstMove == ChessEngine::kNullMove) {
            return false;
        }
        ez_make_move(&pos, firstMove);
        Move secondMove = ChessEngine::uci_to_move(pos, line[1]);
        if (secondMove == ChessEngine::kNullMove) {
            return false;
        }
        if (pos.tiles_[secondMove.to] != ChessEngine::ColoredPiece::NO_COLORED_PIECE) {
            // We want to avoid positions where the second move is a capture, since those are more likely to be tactical and less likely to be in our training distribution.
            return false;
        }
        
        char buf[32];
        
        if (pos.turn_ == ChessEngine::Color::BLACK) {
            score = -score;
        }

        if (pos.fen() == "5rk1/pp1b1ppp/1q2p3/3pP3/1B3Pn1/3B1N2/P3Q1PP/RNr1KR2 b Q - 1 1") {
            std::cout << fen << "\n";
            std::cout << line.size() << line[0] << " " << line[1] << std::endl;
            std::cout << firstMove.uci() << " " << secondMove.uci() << "\n";
            std::cout << "Debug position found! Score: " << score << std::endl;
        }

        snprintf(buf, sizeof(buf), "%+.2f", score);
        out << pos.fen() << "|" << buf << "\n";
        return true;
    } catch (const std::exception& e) {
        // ignore parsing errors and continue
        out << "Error parsing line: " << e.what() << "\n";
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
                            std::cout << "Processed " << num_processed / 1000 << "K positions...\n";
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
