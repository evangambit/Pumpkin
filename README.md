

```
sudo apt-get install -y libgflags-dev libgtest-dev

# Run tests.
g++ -std=c++20 -o test_runner $(find src/ -name "*.cpp" | grep -Ev '(main|uci|make_tables).cpp') -I/usr/local/include -L/usr/local/lib -lgtest -lgtest_main -pthread && ./test_runner

# Run one test
g++ -std=c++20 -o test_runner src/eval/nnue/tests/nnue-tests.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main|make_tables|pgns2fens)\\.cpp") -I/usr/local/include -L/usr/local/lib -lgtest -lgtest_main -pthread && ./test_runner

# Update NNUE object file (model_bin.o) from a binary file

xxd -i model.bin > model_data.c
xxd -i qst.bin > qst_data.c

# Build main
sh build.sh uci src/uci.cpp -O3 -DNDEBUG model_data.c 

./main rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2

# Build uci

g++ -std=c++20 -o uci src/uci.cpp model_bin.o $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main|make_tables)\\.|pgns2fenscpp") -pthread -L/usr/local/lib -lgflags -DNDEBUG -O3

# Make tables

g++ -std=c++20 -o make_tables src/eval/nnue/make_tables.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main|make_tables|pgns2fens)\\.cpp") -pthread -L/usr/local/lib -lgflags -DNDEBUG -O3

g++ -std=c++20 -o nnue_main src/eval/nnue/main.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main|make_tables|pgns2fens)\\.cpp") -pthread -L/usr/local/lib -lgflags && ./nnue_main

# cutechess

cutechess/build/cutechess-cli -engine cmd=uci arg="evaluator nnue" -engine cmd=old arg="evaluator nnue" -each tc=40/60 proto=uci -rounds 100 -debug

```

# pgn2fen

sh build.sh p2f src/pgns2fens.cpp  -O3 -DNDEBUG

Randomly drop 90% of lines (better position diversity).

$ ./p2f --input_path pgns/ | awk 'BEGIN {srand()} rand() <= 0.10' > data/stock/pos.txt

Data comes from https://huggingface.co/datasets/official-stockfish/fishtest_pgns

## Perf Analysis

/usr/local/go/bin/go install github.com/google/pprof@latest

g++ -std=c++20 -o uci src/uci.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main|make_tables|pgns2fens)\\.cpp") -pthread -lgflags -DNDEBUG -O3 $(pkg-config --cflags --libs libprofiler)

CPUPROFILE=/tmp/prof.out ./uci "move e2e4 c7c5 g1f3 d7d6" "go depth 8" "lazyquit"

~/go/bin/pprof -png ./uci /tmp/prof.out

## Known bugs

- Capturing enpassant into check (e.g. "r4Q2/1pk2pp1/8/3qpP1K/8/8/PP5B/n7 w - - 0 25")
