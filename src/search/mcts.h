#ifndef MCTS_H
#define MCTS_H

#include <vector>
#include <atomic>
#include <cmath>
#include "negamax.h"

namespace ChessEngine {

struct MCTSNode {
    uint64_t hash = 0;
    std::atomic<uint32_t> visitCount{0};
    std::atomic<int64_t>  valueSum{0};
    
    MCTSNode() {}
    MCTSNode(uint64_t h, uint32_t v, int64_t s) : hash(h), visitCount(v), valueSum(s) {}
    
    void update(int64_t value) {
        visitCount.fetch_add(1, std::memory_order_relaxed);
        valueSum.fetch_add(value, std::memory_order_relaxed); // Much safer/faster than CAS loop
    }
};

// Global TT (Pre-allocated to prevent rehashing and control memory)
// 1 << 22 is ~4 million entries. Adjust based on your RAM budget.
constexpr size_t TT_SIZE = 1 << 26; 
static std::vector<MCTSNode> mctsTree(TT_SIZE);

inline double calculate_u(float policy, uint64_t parentVisits, uint64_t childVisits) {
    const double C_PUCT = 2.5; 
    return C_PUCT * policy * std::sqrt(parentVisits) / (1 + childVisits);
}

template<Color TURN>
inline Move select_move_puct(Position& pos, uint64_t parentVisits = 1) {
    Move bestMove;
    double bestScore = -1e10;

    ExtMove moves[kMaxNumMoves];
    ExtMove* end = compute_legal_moves<TURN>(&pos, moves);
  
    for (ExtMove* move = moves; move < end; ++move) {
        make_move<TURN>(&pos, move->move);
        const uint64_t hash = pos.currentState_.hash;
        
        // Find child in the fixed-size array
        MCTSNode* child = nullptr;
        if (mctsTree[hash % TT_SIZE].hash == hash) {
            child = &mctsTree[hash % TT_SIZE];
        }
        undo<TURN>(&pos);

        uint32_t childVisits = child ? child->visitCount.load() : 0;
        double q = (child && childVisits > 0) ? 
                   (static_cast<double>(child->valueSum.load()) / childVisits) : 0.0;
        
        double u = calculate_u(0.5f, parentVisits, childVisits); 
        
        if (q + u > bestScore) {
            bestScore = q + u;
            bestMove = move->move;
        }
    }
    return bestMove;
}

inline void mcts(Thread *thread, size_t iterations) {
    for (size_t i = 0; i < iterations; ++i) {
        Position &pos = thread->position_;
        std::vector<MCTSNode*> path;

        // 1. SELECT
        while (true) {
            uint64_t h = pos.currentState_.hash;
            MCTSNode* entry = &mctsTree[h % TT_SIZE];

            // If the slot is empty or holds a different position, it's a leaf/frontier
            if (entry->hash != h) break; 

            path.push_back(entry);
            Move bestMove;
            if (pos.turn_ == Color::WHITE) {
                bestMove = select_move_puct<Color::WHITE>(pos, entry->visitCount.load());
                make_move<Color::WHITE>(&pos, bestMove);
            } else {
                bestMove = select_move_puct<Color::BLACK>(pos, entry->visitCount.load());
                make_move<Color::BLACK>(&pos, bestMove);
            }
        }

        // 2. EVALUATE (Using qsearch)
        int64_t value;
        std::atomic<bool> neverStopThinking{false};
        
        if (pos.turn_ == Color::WHITE) {
            value = qsearch<Color::WHITE>(
              thread,
              ColoredEvaluation<Color::WHITE>(kMinEval),
              ColoredEvaluation<Color::WHITE>(kMaxEval),
              0, 0,
              thread->root_frame(),
              &neverStopThinking
            ).evaluation.value;
        } else {
            value = qsearch<Color::BLACK>(
              thread,
              ColoredEvaluation<Color::BLACK>(kMinEval),
              ColoredEvaluation<Color::BLACK>(kMaxEval),
              0, 0,
              thread->root_frame(),
              &neverStopThinking
            ).evaluation.value;
        }

        // 3. EXPAND
        uint64_t leafHash = pos.currentState_.hash;
        MCTSNode* leafNode = &mctsTree[leafHash % TT_SIZE];
        // Overwrite standard TT logic (you might want to add depth/visit checks here later)
        leafNode->hash = leafHash;
        leafNode->visitCount = 0; // Will be incremented in the backprop step
        leafNode->valueSum = 0;

        for (auto* entry : path) {
            ez_undo(&pos); // Undo the move to backtrack up the tree
        }

        
        // Add leaf to path so it gets updated too
        path.push_back(leafNode);

        // 4. BACKPROPAGATE
        for (auto* entry : path) {
            entry->update(value);
            value = -value; // Flip perspective for the next node up the tree
        }
    }

    ExtMove moves[kMaxNumMoves];
    ExtMove* end;
    if (thread->position_.turn_ == Color::WHITE) {
        end = compute_legal_moves<Color::WHITE>(&thread->position_, moves);
    } else {
        end = compute_legal_moves<Color::BLACK>(&thread->position_, moves);
    }
    for (ExtMove* move = moves; move < end; ++move) {
        ez_make_move(&thread->position_, move->move);
        const uint64_t hash = thread->position_.currentState_.hash;
        MCTSNode* child = nullptr;
        if (mctsTree[hash % TT_SIZE].hash == hash) {
            child = &mctsTree[hash % TT_SIZE];
        }
        if (child != nullptr) {
          move->score = child->valueSum.load() / child->visitCount.load();
        } else {
          move->score = kMinEval;
        }
        ez_undo(&thread->position_);
    }
    std::sort(
      moves,
      end,
      [](const ExtMove& a, const ExtMove& b) {
        return a.score > b.score;
      }
    );
    for (ExtMove* move = moves; move < end; ++move) {
        ez_make_move(&thread->position_, move->move);
        const uint64_t hash = thread->position_.currentState_.hash;
        MCTSNode* child = nullptr;
        if (mctsTree[hash % TT_SIZE].hash == hash) {
            child = &mctsTree[hash % TT_SIZE];
        }
        ez_undo(&thread->position_);

        uint32_t childVisits = child ? child->visitCount.load() : 0;
        double q = (child && childVisits > 0) ? 
                   (static_cast<double>(child->valueSum.load()) / childVisits) : 0.0;
        
        if (q > -1e10) { // Only consider moves that were visited
            std::cout << "Move: " << move->move.uci() << " Visits: " << childVisits << " Q-value: " << q << std::endl;
        }
    }
}

}  // namespace ChessEngine
#endif