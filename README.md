

```
sudo apt-get install -y libgflags-dev libgtest-dev

# Run tests.
sh build.sh test_runner $(find src -name "*.cpp" | grep 'Tests\.cpp') -lgtest -lgtest_main -pthread && ./test_runner

# Run one test
sh build.sh test_runner -lgtest src/eval/nnue/tests/nnue-tests.cpp -lgtest_main && ./test_runner

# Update NNUE object file (model_bin.o) from a binary file

    xxd -i model.bin > model_data.cpp
    xxd -i qst.bin > qst_data.cpp

# Build uci

    ./build.sh uci src/uci/main.cpp -O3 -DNDEBUG

# Make tables

    ./build.sh mt src/eval/MakeTablesMain.cpp -O3 -DNDEBUG

# cutechess

With time control (40 moves / 60 seconds):

    ~/bin/cutechess-cli \
    -engine cmd=uci name=NewNNUE arg="evaluator nnue" \
    -engine cmd=old name=OldNNUE arg="evaluator nnue" \
    -each tc=40/60 proto=uci \
    -rounds 10 \
    -concurrency 8 \
    -pgnout tournament/a.pgn \
    -openings file=/Users/morganredding/Downloads/Unique_110225/Unique_v110225.pgn plies=12

With nodes/move

    ~/bin/cutechess-cli \
    -engine cmd=uci name=NewNNUE arg="evaluator nnue" \
    -engine cmd=old name=OldNNUE arg="evaluator nnue" \
    -each nodes=1000000 tc=inf proto=uci \
    -rounds 100 \
    -concurrency 6 \
    -pgnout tournament/a.pgn \
    -openings file=/Users/morganredding/Downloads/Unique_110225/Unique_v110225.pgn plies=12

```

# pgn2fen

    sh build.sh p2f src/PgnsToFensMain.cpp  -O3 -DNDEBUG

Randomly drop 90% of lines (better position diversity).

    $ ./p2f --input_path pgns/ | awk 'BEGIN {srand()} rand() <= 0.10' > data/stock/pos.txt

Data comes from https://huggingface.co/datasets/official-stockfish/fishtest_pgns

## Perf Analysis

/usr/local/go/bin/go install github.com/google/pprof@latest

    sh build.sh uci src/uci.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main|make_tables|pgns2fens)\\.cpp") -DNDEBUG -O3 $(pkg-config --cflags --libs libprofiler)

    CPUPROFILE=/tmp/prof.out ./uci "move e2e4 c7c5 g1f3 d7d6" "go depth 8" "lazyquit"

    ~/go/bin/pprof -png ./uci /tmp/prof.out

## Known bugs

- Capturing enpassant into check (e.g. "r4Q2/1pk2pp1/8/3qpP1K/8/8/PP5B/n7 w - - 0 25")
