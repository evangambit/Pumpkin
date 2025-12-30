

```
# Run tests.
g++ -std=c++20 -o test_runner $(find src/ -name "*.cpp" | grep -v main.cpp) \
-I/usr/local/include -L/usr/local/lib -lgtest -lgtest_main -pthread && ./test_runner


# Build main
g++ -std=c++20 -o main $(find src/ -name "*.cpp" | grep -Ev "[Tt]ests?\\.cpp") -pthread

./main rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2
```
