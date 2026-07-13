
# Build

Don't reuse one `build/` directory across WSL and Windows — CMake caches absolute paths. Use separate trees (e.g. `build` in WSL, `build-win` on Windows).

## Linux / WSL

```bash
sudo apt-get install -y cmake g++ libgtest-dev

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPUMPKIN_DEFAULT_EVALUATOR=n
cmake --build build --target uci
./build/uci
```

## Windows (MSVC)

```powershell
cmake -S . -B build-win -DPUMPKIN_DEFAULT_EVALUATOR=n
cmake --build build-win --config Release --target uci
.\build-win\Release\uci.exe
```

Tests are built when GTest is available (`libgtest-dev` on Linux). Without GTest, configure still succeeds and only skips tests.

# Run tests

    ctest --test-dir build --output-on-failure

# Run one test file
    ./build/test_runner --gtest_filter='NnueTests.*'
    ./build/test_runner --gtest_filter='ByHandTests.*'
    ./build/test_runner --gtest_filter='GeometryTests.*'
    ./build/test_runner --gtest_filter='SearchTests.*'

# Update NNUE object file from a binary file

    xxd -i model.bin > model_data.cpp
    xxd -i qst.bin > qst_data.cpp

# Build uci

    cmake --build build --target uci
    ./build/uci

# cutechess

With time control (40 moves / 60 seconds):

    ~/bin/cutechess-cli \
    -engine cmd=uci name=NewNNUE arg="evaluator nnue model.bin" \
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

Randomly drop 90% of lines (better position diversity).

    $ ./p2f --input_path pgns/ | awk 'BEGIN {srand()} rand() <= 0.10' > data/stock/pos.txt

Data comes from https://huggingface.co/datasets/official-stockfish/fishtest_pgns

# Perf Analysis

/usr/local/go/bin/go install github.com/google/pprof@latest

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-g $(pkg-config --cflags --libs libprofiler)"
    cmake --build build --target uci

    CPUPROFILE=/tmp/prof.out ./build/uci "evaluator nnue" "move e2e4 c7c5 g1f3 d7d6" "go depth 8" "lazyquit"

    ~/go/bin/pprof -png ./build/uci /tmp/prof.out

todo: make this nicer


# "-g" allows per-line profiling
    cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-g -L$(brew --prefix gperftools)/lib -lprofiler"
    cmake --build build --target uci
    CPUPROFILE=/tmp/prof.out ./build/uci "evaluator nnue" "move e2e4 c7c5 g1f3 d7d6" "go depth 7" "lazyquit"

    ~/go/bin/pprof -top ./build/uci /tmp/prof.out

    ~/go/bin/pprof -list _evaluate ./build/uci /tmp/prof.out

# Build dataset

cd nnue/
python setup.py build_ext --inplace --force
cd ..
