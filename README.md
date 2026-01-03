

```
sudo apt-get install -y libgflags-dev libgtest-dev

# Run tests.
g++ -std=c++20 -o test_runner $(find src/ -name "*.cpp" | grep -Ev '(main|uci).cpp') -I/usr/local/include -L/usr/local/lib -lgtest -lgtest_main -pthread && ./test_runner

# Run one test
g++ -std=c++20 -o test_runner src/game/movegen/tests/movegen-tests.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main)\\.cpp") -I/usr/local/include -L/usr/local/lib -lgtest -lgtest_main -pthread && ./test_runner


# Build main
g++ -std=c++20 -o main src/main.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main)\\.cpp") -pthread -L/usr/local/lib -lgflags

./main rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2

# Build uci

g++ -std=c++20 -o uci src/uci.cpp $(find src/ -name "*.cpp" | grep -Ev "([Tt]ests?|uci|main)\\.cpp") -pthread -L/usr/local/lib -lgflags
```

bug:

 ./main --fen 'r1bq1bnr/pp1k1ppp/4p3/2pP4/6Q1/8/PPPP1PPP/RNB1K1NR b KQ - 0 6'
