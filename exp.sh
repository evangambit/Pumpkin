
declare -a values=("50" "100" "150")

for i in "${values[@]}"
do
  sh build.sh uci-${i} src/uci/main.cpp -DNDEBUG -O3 -DPARAM=${i}
done

engines=()
for i in "${values[@]}"
do
  engines+=(-engine cmd="./uci-${i} \"evaluator nnue\" \"setoption name Overhead value 10\"" name="uci-${i}")
done

../c-chess-cli/c-chess-cli \
"${engines[@]}" \
-each tc=3.0+0.03 option.Hash=2048 \
-games 1600 -repeat -concurrency 14 -openings file=../UHO_4060_v4.epd order=random \
-pgn scaling_results.pgn 3 -resign count=3 score=700 -draw number=40
