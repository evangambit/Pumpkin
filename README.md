

```
# Run tests.
g++ -std=c++20 -o test_runner $(find src -name "*[Tt]est*.cpp") src/game/*.cpp src/string_utils.cpp -I/usr/local/include -L/usr/local/lib -lgtest -lgtest_main -pthread && ./test_runner
```
