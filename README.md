

g++ -std=c++20 -o test_runner src/game/tests/PositionTests.cpp src/game/*.cpp src/string_utils.cpp -I/usr/local/include -L/usr/local/lib -lgtest -lgtest_main -pthread && ./test_runner

g++ -std=c++20 -o test_runner src/game/movegen/tests/movegen-tests.cpp src/game/*.cpp src/string_utils.cpp -I/usr/local/include -L/usr/local/lib -lgtest -lgtest_main -pthread && ./test_runner
