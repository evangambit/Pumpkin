

```
sudo apt-get install -y libgflags-dev libgtest-dev

# Run tests.
g++ -std=c++20 -o test_runner $(find src/ -name "*.cpp" | grep -Ev '(main|uci|make_tables).cpp') -I/usr/local/include -L/usr/local/lib -lgtest -lgtest_main -pthread && ./test_runner

# Run one test
g++ -std=c++20 -o test_runner src/game/tests/PositionTests.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main)\\.cpp") -I/usr/local/include -L/usr/local/lib -lgtest -lgtest_main -pthread && ./test_runner


# Build main
g++ -std=c++20 -o main src/main.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main|make_tables)\\.cpp") -pthread -L/usr/local/lib -lgflags

./main rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2

# Build uci

g++ -std=c++20 -o uci src/uci.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main|make_tables)\\.cpp") -pthread -L/usr/local/lib -lgflags -DNDEBUG -O3

# Make tables

g++ -std=c++20 -o make_tables src/eval/nnue/make_tables.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main|make_tables)\\.cpp") -pthread -L/usr/local/lib -lgflags -DNDEBUG -O3

g++ -std=c++20 -o nnue_main src/eval/nnue/main.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main|make_tables)\\.cpp") -pthread -L/usr/local/lib -lgflags && ./nnue_main

# cutechess

cutechess/build/cutechess-cli -engine cmd=uci -engine cmd=uci -each tc=40/60 proto=uci -rounds 1 -debug^C

```

## Known bugs

- Capturing enpassant into check (e.g. "r4Q2/1pk2pp1/8/3qpP1K/8/8/PP5B/n7 w - - 0 25")
