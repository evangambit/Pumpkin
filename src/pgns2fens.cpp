/**
 * pgns2fens - Extract quiet positions from PGN files
 *
 * Recursively traverses a directory of PGN files and outputs FEN strings
 * for "quiet" positions. A position is quiet if the next move played
 * is not a check, capture, or promotion.
 *
 * Usage: pgns2fens --input_path=<directory> [--skip_percentage=0.9]
 *
 * Output: One FEN string per line to stdout.
 * Errors/warnings are written to stderr.
 *
 * By default, randomly skips 90% of positions to avoid including too many
 * highly similar positions. The skip percentage can be controlled with
 * --skip_percentage flag (0.0 = skip none, 1.0 = skip all).
 *
 * Example commands:
 * ./p2f --input_path pgns/ > data/pos.txt
 * ./p2f --input_path pgns/ --skip_percentage 0.8 > data/pos.txt
 * terashuf < data/pos.txt > data/pos.shuf.txt
 * ./make_tables data/pos.shuf.txt data/nnue
 * ./qst_make_tables data/pos.shuf.txt data/qst
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include <algorithm>
#include <random>
#include <queue>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>
#include <zlib.h>
#include <gflags/gflags.h>

#include "game/Position.h"
#include "game/geometry.h"
#include "game/movegen/movegen.h"
#include "string_utils.h"

// Command line flags
DEFINE_string(input_path, "", "Directory or file path containing PGN files to process");
DEFINE_int32(max_games, -1, "Maximum number of games to process (-1 for unlimited)");
DEFINE_bool(verbose, false, "Enable verbose output");
DEFINE_bool(include_eval, true, "Include evaluation comments in output");
DEFINE_bool(include_moves, false, "Include the best move (and up to 9 random other moves)");
DEFINE_double(skip_percentage, 0.9, "Percentage of positions to skip randomly (0.0 = skip none, 0.9 = skip 90%)");
DEFINE_uint64(dedup_cache_size, 10000, "Size of LRU cache for deduplicating positions (0 = disabled)");
DEFINE_int32(num_threads, 0, "Number of threads to use (0 = auto-detect based on hardware)");

namespace fs = std::filesystem;
using namespace ChessEngine;

// Global counter for tracking processed games (atomic for thread safety)
static std::atomic<int> g_totalGamesProcessed{0};
static std::atomic<int> g_lastProgressUpdate{0};
static const int PROGRESS_INTERVAL = 100000; // Update every 100k games

// Mutex for synchronized stdout output
static std::mutex g_outputMutex;
static std::mutex g_progressMutex;

// Thread-local random number generator for position sampling
static thread_local std::mt19937 t_gen{std::random_device{}()};
static thread_local std::uniform_real_distribution<double> t_dist(0.0, 1.0);

// Thread-local LRU deduplication cache
static thread_local std::unordered_set<uint64_t> t_seenHashes;
static thread_local std::queue<uint64_t> t_hashOrder;

// Returns true if position is new (not a duplicate), and adds it to the cache
static bool check_and_insert_position(uint64_t hash) {
  if (FLAGS_dedup_cache_size == 0) {
    return true;  // Deduplication disabled
  }
  
  if (t_seenHashes.count(hash)) {
    return false;  // Duplicate
  }
  
  // Insert new hash
  t_seenHashes.insert(hash);
  t_hashOrder.push(hash);
  
  // Evict oldest if over capacity
  while (t_hashOrder.size() > FLAGS_dedup_cache_size) {
    uint64_t oldest = t_hashOrder.front();
    t_hashOrder.pop();
    t_seenHashes.erase(oldest);
  }
  
  return true;
}

/**
 * Print progress update if we've processed another 100k games
 */
void check_and_print_progress() {
  int current = g_totalGamesProcessed.load();
  int lastUpdate = g_lastProgressUpdate.load();
  if (current - lastUpdate >= PROGRESS_INTERVAL) {
    // Try to claim the update
    if (g_lastProgressUpdate.compare_exchange_strong(lastUpdate, current)) {
      std::lock_guard<std::mutex> lock(g_progressMutex);
      std::cerr << "Progress: Processed " << current << " games..." << std::endl;
    }
  }
}

/**
 * Checks if a string looks like a move number (e.g., "1.", "12.", "1...")
 */
bool is_move_number(const std::string& token) {
  if (token.empty()) return false;
  size_t i = 0;
  while (i < token.size() && std::isdigit(token[i])) {
    i++;
  }
  if (i == 0) return false;
  // Rest should be dots
  while (i < token.size() && token[i] == '.') {
    i++;
  }
  return i == token.size();
}

/**
 * Checks if a token is a game result.
 */
bool is_result(const std::string& token) {
  return token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*";
}

/**
 * Checks if a token is a NAG (Numeric Annotation Glyph) like $1, $2, etc.
 */
bool is_nag(const std::string& token) {
  return !token.empty() && token[0] == '$';
}

/**
 * Parses a SAN move and returns the corresponding Move, or kNullMove if invalid.
 * Uses the existing uci_to_move function by first converting SAN to UCI.
 */
Move san_to_move(const Position& pos, const std::string& san) {
  // Clean up the SAN: remove check/mate indicators and annotations
  std::string cleanSan = san;
  while (!cleanSan.empty() && (cleanSan.back() == '+' || cleanSan.back() == '#'
         || cleanSan.back() == '!' || cleanSan.back() == '?')) {
    cleanSan.pop_back();
  }

  if (cleanSan.empty()) {
    return kNullMove;
  }

  // Generate all legal moves and match
  ExtMove moves[kMaxNumMoves];
  ExtMove* end;
  Position tempPos = pos;

  if (pos.turn_ == Color::WHITE) {
    end = compute_legal_moves<Color::WHITE>(&tempPos, moves);
  } else {
    end = compute_legal_moves<Color::BLACK>(&tempPos, moves);
  }

  // Handle castling
  if (cleanSan == "O-O" || cleanSan == "0-0") {
    for (ExtMove* m = moves; m != end; ++m) {
      if (m->move.moveType == MoveType::CASTLE) {
        // Kingside castle: king moves right
        if (m->move.to > m->move.from) {
          return m->move;
        }
      }
    }
    return kNullMove;
  }
  if (cleanSan == "O-O-O" || cleanSan == "0-0-0") {
    for (ExtMove* m = moves; m != end; ++m) {
      if (m->move.moveType == MoveType::CASTLE) {
        // Queenside castle: king moves left
        if (m->move.to < m->move.from) {
          return m->move;
        }
      }
    }
    return kNullMove;
  }

  // Parse the SAN move
  Piece piece = Piece::PAWN;
  char promoPiece = 0;
  char destFile = 0, destRank = 0;
  char srcFile = 0, srcRank = 0;

  size_t idx = 0;

  // First character might be piece type
  if (!cleanSan.empty() && std::isupper(cleanSan[0]) && cleanSan[0] != 'O') {
    switch (cleanSan[0]) {
      case 'N': piece = Piece::KNIGHT; break;
      case 'B': piece = Piece::BISHOP; break;
      case 'R': piece = Piece::ROOK; break;
      case 'Q': piece = Piece::QUEEN; break;
      case 'K': piece = Piece::KING; break;
      default: return kNullMove;
    }
    idx++;
  }

  // Check for promotion at end (e.g., e8=Q)
  for (size_t i = idx; i < cleanSan.size(); i++) {
    if (cleanSan[i] == '=') {
      if (i + 1 < cleanSan.size()) {
        promoPiece = cleanSan[i + 1];
      }
      cleanSan = cleanSan.substr(0, i);
      break;
    }
  }

  // Also handle promotion without '=' (e.g., e8Q)
  if (promoPiece == 0 && cleanSan.size() >= 2) {
    char lastChar = cleanSan.back();
    if (lastChar == 'Q' || lastChar == 'R' || lastChar == 'B' || lastChar == 'N') {
      char secondLast = cleanSan[cleanSan.size() - 2];
      if (secondLast == '1' || secondLast == '8') {
        promoPiece = lastChar;
        cleanSan.pop_back();
      }
    }
  }

  // Now parse disambiguation and destination
  // Format can be: e4, Nf3, Nge2, N1e2, Ng1e2, exd5, etc.
  std::string rest = cleanSan.substr(idx);

  // Remove 'x' for captures
  rest.erase(std::remove(rest.begin(), rest.end(), 'x'), rest.end());

  if (rest.size() < 2) {
    return kNullMove;
  }

  // Last two chars are always destination
  destFile = rest[rest.size() - 2];
  destRank = rest[rest.size() - 1];

  if (destFile < 'a' || destFile > 'h' || destRank < '1' || destRank > '8') {
    return kNullMove;
  }

  // Disambiguation: anything before destination
  std::string disambig = rest.substr(0, rest.size() - 2);
  for (char c : disambig) {
    if (c >= 'a' && c <= 'h') {
      srcFile = c;
    } else if (c >= '1' && c <= '8') {
      srcRank = c;
    }
  }

  // Convert destination to square index
  int destX = destFile - 'a';
  int destY = '8' - destRank;
  SafeSquare destSq = SafeSquare(destY * 8 + destX);

  // Find matching move
  for (ExtMove* m = moves; m != end; ++m) {
    if (m->piece != piece) continue;
    if (m->move.to != destSq) continue;

    // Check disambiguation
    int fromX = m->move.from % 8;
    int fromY = m->move.from / 8;
    char fromFile = 'a' + fromX;
    char fromRank = '8' - fromY;

    if (srcFile && fromFile != srcFile) continue;
    if (srcRank && fromRank != srcRank) continue;

    // Check promotion
    if (promoPiece) {
      if (m->move.moveType != MoveType::PROMOTION) continue;
      int promoIdx = m->move.promotion;  // 0=N, 1=B, 2=R, 3=Q
      char expectedPromo = "NBRQ"[promoIdx];
      if (expectedPromo != promoPiece) continue;
    } else {
      if (m->move.moveType == MoveType::PROMOTION) continue;
    }

    return m->move;
  }

  return kNullMove;
}

/**
 * Finds the ExtMove for a given Move from the list of legal moves.
 */
ExtMove find_ext_move(Position& pos, Move move) {
  ExtMove moves[kMaxNumMoves];
  ExtMove* end;

  if (pos.turn_ == Color::WHITE) {
    end = compute_legal_moves<Color::WHITE>(&pos, moves);
  } else {
    end = compute_legal_moves<Color::BLACK>(&pos, moves);
  }

  for (ExtMove* m = moves; m != end; ++m) {
    if (m->move == move) {
      return *m;
    }
  }
  return kNullExtMove;
}

/**
 * Checks if a move gives check.
 */
bool gives_check(Position& pos, Move move) {
  // Remember who is making the move
  Color mover = pos.turn_;
  
  // Make the move
  if (mover == Color::WHITE) {
    make_move<Color::WHITE>(&pos, move);
  } else {
    make_move<Color::BLACK>(&pos, move);
  }

  // After the move, it's the opponent's turn. Check if their king is attacked.
  bool inCheck;
  if (pos.turn_ == Color::WHITE) {
    SafeSquare kingSq = lsb_i_promise_board_is_not_empty(
        pos.pieceBitboards_[ColoredPiece::WHITE_KING]);
    inCheck = can_enemy_attack<Color::WHITE>(pos, kingSq);
  } else {
    SafeSquare kingSq = lsb_i_promise_board_is_not_empty(
        pos.pieceBitboards_[ColoredPiece::BLACK_KING]);
    inCheck = can_enemy_attack<Color::BLACK>(pos, kingSq);
  }

  if (mover == Color::WHITE) {
    undo<Color::WHITE>(&pos);
  } else {
    undo<Color::BLACK>(&pos);
  }

  return inCheck;
}

/**
 * Checks if a move is "quiet" (not a capture, promotion, or check).
 */
bool is_quiet_move(Position& pos, Move move, const ExtMove& extMove) {
  // Check capture
  if (extMove.capture != ColoredPiece::NO_COLORED_PIECE) {
    return false;
  }

  // Check en passant (which is also a capture)
  if (move.moveType == MoveType::EN_PASSANT) {
    return false;
  }

  // Check promotion
  if (move.moveType == MoveType::PROMOTION) {
    return false;
  }

  // Check if gives check
  if (gives_check(pos, move)) {
    return false;
  }

  return true;
}

/**
 * A token from PGN movetext, possibly with an associated evaluation.
 */
struct PgnToken {
  std::string text;
  std::string eval;  // Empty if no eval comment follows
};

/**
 * Extracts the evaluation from a comment like "{+0.48/20 1.4s}".
 * Returns the portion before the "/", or empty string if not found.
 */
std::string extract_eval_from_comment(const std::string& comment) {
  // Look for pattern: optional sign, digits, optional decimal
  // e.g., "+0.48", "-1.23", "0.00", "#5", "#-3"
  size_t start = 0;
  
  // Skip leading whitespace
  while (start < comment.size() && std::isspace(comment[start])) {
    start++;
  }
  
  if (start >= comment.size()) {
    return "";
  }
  
  // Check for mate score (#N)
  if (comment[start] == '#') {
    size_t end = start + 1;
    bool isNegative = false;
    if (end < comment.size() && (comment[end] == '+' || comment[end] == '-')) {
      if (comment[end] == '-') {
        isNegative = true;
      }
      end++;
    }
    while (end < comment.size() && std::isdigit(comment[end])) {
      end++;
    }
    if (end > start + 1) {
      return isNegative ? "-99" : "99";
    }
    return "";
  }
  
  // Check for numeric eval with optional sign
  size_t end = start;
  if (comment[end] == '+' || comment[end] == '-') {
    end++;
  }
  
  // Must have at least one digit
  if (end >= comment.size() || !std::isdigit(comment[end])) {
    return "";
  }
  
  while (end < comment.size() && std::isdigit(comment[end])) {
    end++;
  }
  
  // Optional decimal part
  if (end < comment.size() && comment[end] == '.') {
    end++;
    while (end < comment.size() && std::isdigit(comment[end])) {
      end++;
    }
  }
  
  // Should be followed by '/' for depth
  if (end < comment.size() && comment[end] == '/') {
    return comment.substr(start, end - start);
  }
  
  return "";
}

/**
 * Tokenizes the movetext section of a PGN, preserving eval comments.
 */
std::vector<PgnToken> tokenize_movetext(const std::string& text) {
  std::vector<PgnToken> tokens;
  std::string current;
  std::string pendingEval;

  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    
    // Handle comments - extract eval if present
    if (c == '{') {
      // Save current token if any
      if (!current.empty()) {
        tokens.push_back({current, ""});
        current.clear();
      }
      
      // Find end of comment
      size_t commentStart = i + 1;
      int depth = 1;
      size_t j = i + 1;
      while (j < text.size() && depth > 0) {
        if (text[j] == '{') depth++;
        else if (text[j] == '}') depth--;
        j++;
      }
      std::string comment = text.substr(commentStart, j - commentStart - 1);
      std::string eval = extract_eval_from_comment(comment);
      
      // Attach eval to previous token if it exists and has no eval yet
      if (!eval.empty() && !tokens.empty() && tokens.back().eval.empty()) {
        tokens.back().eval = eval;
      }
      
      i = j - 1;  // -1 because loop will increment
      continue;
    }
    
    // Skip semicolon comments to end of line
    if (c == ';') {
      if (!current.empty()) {
        tokens.push_back({current, ""});
        current.clear();
      }
      while (i < text.size() && text[i] != '\n') {
        i++;
      }
      continue;
    }
    
    if (std::isspace(c) || c == '(' || c == ')') {
      if (!current.empty()) {
        tokens.push_back({current, ""});
        current.clear();
      }
    } else {
      current += c;
    }
  }
  if (!current.empty()) {
    tokens.push_back({current, ""});
  }

  return tokens;
}

/**
 * Processes a single PGN game's movetext and outputs quiet positions.
 * If startFen is empty, uses the standard starting position.
 * Output is written to the provided stream (for thread-safe buffering).
 */
void process_game(const std::string& movetext, const std::string& startFen, const std::string& filename, std::ostream& out) {
  Position pos = startFen.empty() ? Position::init() : Position(startFen);
  std::vector<PgnToken> tokens = tokenize_movetext(movetext);
  if (tokens.size() < 10) {
    // Typically something went wrong (e.g. "Internal Server Error", "The server encountered an unexpected internal server error", etc.).
    std::cerr << "Warning: Game in " << filename << " has too few tokens (" << tokens.size() << "). Skipping." << std::endl;
    return;
  }
  
  // Track last two moves for output
  Move secondToLastMove = kNullMove;
  Move lastMove = kNullMove;

  for (const PgnToken& token : tokens) {
    // Skip move numbers, results, and NAGs
    if (is_move_number(token.text) || is_result(token.text) || is_nag(token.text)) {
      continue;
    }

    // Try to parse as a move
    Move move = san_to_move(pos, token.text);
    if (move == kNullMove) {
      // Could be a malformed move or unrecognized token
      std::cerr << "Warning: Could not parse move '" << token.text
                << "' in " << filename << std::endl;
      std::cerr << "Starting FEN: " << (startFen.empty() ? "(standard)" : startFen) << std::endl;
      std::cerr << "Current position FEN: " << pos.fen() << std::endl;
      std::cerr << "Full movetext: " << movetext << std::endl;
      exit(1);
    }

    // Find the ExtMove to get capture information
    ExtMove extMove = find_ext_move(pos, move);
    if (extMove.move == kNullMove) {
      std::cerr << "Warning: Could not find ExtMove for '" << token.text
                << "' in " << filename << std::endl;
      exit(1);
    }

    // Check if the position is quiet (move is not capture/promotion/check)
    // Only output if we don't need eval or we have eval data
    // Also randomly skip positions based on skip_percentage flag
    if (is_quiet_move(pos, move, extMove) && (!FLAGS_include_eval || !token.eval.empty())) {
      // Randomly skip positions based on skip_percentage
      if (t_dist(t_gen) >= FLAGS_skip_percentage) {
        // Check for duplicate position before outputting
        if (check_and_insert_position(pos.currentState_.hash)) {
        // Output this position (only keep 1 - skip_percentage of positions)
        out << pos.fen();
      if (FLAGS_include_eval && !token.eval.empty()) {
        out << "|" << token.eval;
      }
      if (FLAGS_include_moves) {
        ExtMove legalMoves[kMaxNumMoves];
        ExtMove* end;
        Position tempPos = pos;
        if (tempPos.turn_ == Color::WHITE) {
          end = compute_legal_moves<Color::WHITE>(&tempPos, legalMoves);
        } else {
          end = compute_legal_moves<Color::BLACK>(&tempPos, legalMoves);
        }
        int numLegalMoves = end - legalMoves;

        // shuffle legal moves.
        std::shuffle(legalMoves, end, t_gen);

        {
          int from = int(move.from);
          int to = int(move.to);
          if (pos.turn_ == Color::BLACK) {
            // Flip the move coordinates for black.
            from = (7 - from / 8) * 8 + (7 - from % 8);
            to = (7 - to / 8) * 8 + (7 - to % 8);
          }
          from += (extMove.piece - 1) * 64;
          to += (extMove.piece - 1) * 64;
          out << "|" << from << "|" << to;
        }

        bool moveFound = false;
        constexpr size_t maxMovesToShow = 10;
        size_t numMovesPrinted = 1;
        for (ExtMove* m = legalMoves; m != end; ++m) {
          if (m->move == move) {
            moveFound = true;
            continue;
          }
          if (numMovesPrinted < maxMovesToShow) {
            int from = int(m->move.from);
            int to = int(m->move.to);
            if (pos.turn_ == Color::BLACK) {
              // Flip the move coordinates for black.
              from = (7 - from / 8) * 8 + (7 - from % 8);
              to = (7 - to / 8) * 8 + (7 - to % 8);
            }
            from += (m->piece - 1) * 64;
            to += (m->piece - 1) * 64;
            out << "|" << from << "|" << to;
            numMovesPrinted++;
          }
        }
        while (numMovesPrinted < maxMovesToShow) {
          out << "|384|384";  // padding for missing moves
          numMovesPrinted++;
        }
        if (!moveFound) {
          std::cerr << "Warning: Move not found in legal moves for '" << token.text
                    << "' in " << filename << std::endl;
          exit(1);
        }
      }
      out << '\n';
      }
      }
    }

    // Update move history
    secondToLastMove = lastMove;
    lastMove = move;

    // Make the move
    if (pos.turn_ == Color::WHITE) {
      make_move<Color::WHITE>(&pos, move);
    } else {
      make_move<Color::BLACK>(&pos, move);
    }
  }
}

/**
 * Reads a gzip-compressed file into a string.
 */
std::string read_gzip_file(const fs::path& filepath) {
  gzFile gz = gzopen(filepath.string().c_str(), "rb");
  if (!gz) {
    std::cerr << "Error: Could not open gzip file " << filepath << std::endl;
    return "";
  }

  std::string content;
  char buffer[8192];
  int bytesRead;

  while ((bytesRead = gzread(gz, buffer, sizeof(buffer))) > 0) {
    content.append(buffer, bytesRead);
  }

  if (bytesRead < 0) {
    int errnum;
    const char* errMsg = gzerror(gz, &errnum);
    std::cerr << "Error reading gzip file " << filepath << ": " << errMsg << std::endl;
  }

  gzclose(gz);
  return content;
}

/**
 * Helper to process a completed game and reset state.
 * Returns true if we should stop processing (max games reached).
 * Output is written to the provided stream.
 */
bool finish_game(std::string& movetext, std::string& startFen, const std::string& filepath, int& gameCount, std::ostream& out) {
  if (movetext.empty()) {
    return false;
  }
  
  if (FLAGS_max_games == -1 || g_totalGamesProcessed.load() < FLAGS_max_games) {
    process_game(movetext, startFen, filepath, out);
    gameCount++;
    g_totalGamesProcessed.fetch_add(1);
    check_and_print_progress();
  }
  
  movetext.clear();
  startFen.clear();
  
  if (FLAGS_max_games != -1 && g_totalGamesProcessed.load() >= FLAGS_max_games) {
    if (FLAGS_verbose) {
      std::lock_guard<std::mutex> lock(g_progressMutex);
      std::cerr << "Reached maximum games limit (" << FLAGS_max_games << ")" << std::endl;
    }
    return true;
  }
  return false;
}

/**
 * Processes a PGN file (possibly gzip-compressed).
 * Uses a local buffer for thread-safe output.
 */
void process_pgn_file(const fs::path& filepath) {
  std::string content;
  std::string ext = filepath.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

  if (ext == ".gz") {
    content = read_gzip_file(filepath);
    if (content.empty()) {
      return;
    }
  } else {
    std::ifstream file(filepath);
    if (!file.is_open()) {
      std::lock_guard<std::mutex> lock(g_progressMutex);
      std::cerr << "Error: Could not open " << filepath << std::endl;
      return;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    content = ss.str();
  }

  // Thread-local output buffer
  std::ostringstream outputBuffer;

  std::istringstream stream(content);
  std::string line;
  std::string movetext;
  std::string startFen;
  bool inMovetext = false;
  int gameCount = 0;

  while (std::getline(stream, line)) {
    // Check for tag line
    if (!line.empty() && line[0] == '[') {
      // If we were collecting movetext, process the previous game
      if (inMovetext) {
        if (finish_game(movetext, startFen, filepath.string(), gameCount, outputBuffer)) {
          break;
        }
        inMovetext = false;
      }
      
      // Check for FEN tag: [FEN "..."]
      if (line.size() > 6 && line.substr(0, 5) == "[FEN ") {
        size_t firstQuote = line.find('"');
        size_t lastQuote = line.rfind('"');
        if (firstQuote != std::string::npos && lastQuote != std::string::npos && lastQuote > firstQuote) {
          startFen = line.substr(firstQuote + 1, lastQuote - firstQuote - 1);
          // Append missing halfmove/fullmove counters if needed (FEN should have 6 parts)
          int spaceCount = std::count(startFen.begin(), startFen.end(), ' ');
          if (spaceCount == 3) {
            startFen += " 0 1";
          } else if (spaceCount == 4) {
            startFen += " 1";
          }
        }
      }
      continue;
    }

    // Skip empty lines at the start
    ltrim(&line);
    if (line.empty()) {
      if (inMovetext) {
        if (finish_game(movetext, startFen, filepath.string(), gameCount, outputBuffer)) {
          break;
        }
        inMovetext = false;
      }
      continue;
    }

    // We're in movetext
    inMovetext = true;
    movetext += " " + line;
  }

  // Process any remaining game
  finish_game(movetext, startFen, filepath.string(), gameCount, outputBuffer);

  // Flush buffer to stdout under mutex
  {
    std::lock_guard<std::mutex> lock(g_outputMutex);
    std::cout << outputBuffer.str();
  }

  if (FLAGS_verbose) {
    std::lock_guard<std::mutex> lock(g_progressMutex);
    std::cerr << "Processed " << gameCount << " games from " << filepath << std::endl;
  }
}

/**
 * Checks if a path is a PGN file (either .pgn or .pgn.gz).
 */
bool is_pgn_file(const fs::path& filepath) {
  std::string filename = filepath.filename().string();
  std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
  return filename.size() > 4 && 
         (filename.substr(filename.size() - 4) == ".pgn" ||
          (filename.size() > 7 && filename.substr(filename.size() - 7) == ".pgn.gz"));
}

/**
 * Worker function for processing files from a shared queue.
 */
void worker_thread(std::vector<fs::path>& files, std::atomic<size_t>& fileIndex) {
  while (true) {
    // Check if max games reached
    if (FLAGS_max_games != -1 && g_totalGamesProcessed.load() >= FLAGS_max_games) {
      break;
    }
    
    // Get next file index atomically
    size_t idx = fileIndex.fetch_add(1);
    if (idx >= files.size()) {
      break;
    }
    
    const fs::path& filepath = files[idx];
    if (FLAGS_verbose) {
      std::lock_guard<std::mutex> lock(g_progressMutex);
      std::cerr << "Processing file: " << filepath << std::endl;
    }
    
    process_pgn_file(filepath);
  }
}

/**
 * Recursively finds and processes all PGN files in a directory.
 * Uses multiple threads for parallel processing.
 */
void process_directory(const fs::path& dirpath) {
  // Collect all PGN files first
  std::vector<fs::path> files;
  for (const auto& entry : fs::recursive_directory_iterator(dirpath)) {
    if (entry.is_regular_file() && is_pgn_file(entry.path())) {
      files.push_back(entry.path());
    }
  }
  
  if (files.empty()) {
    std::cerr << "No PGN files found in " << dirpath << std::endl;
    return;
  }
  
  // Determine number of threads
  int numThreads = FLAGS_num_threads;
  if (numThreads <= 0) {
    numThreads = static_cast<int>(std::thread::hardware_concurrency());
    if (numThreads <= 0) numThreads = 1;
  }
  // Don't use more threads than files
  numThreads = std::min(numThreads, static_cast<int>(files.size()));
  
  std::cerr << "Found " << files.size() << " PGN files, processing with " 
            << numThreads << " threads..." << std::endl;
  
  // Shared file index for work stealing
  std::atomic<size_t> fileIndex{0};
  
  // Launch worker threads
  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for (int i = 0; i < numThreads; ++i) {
    threads.emplace_back(worker_thread, std::ref(files), std::ref(fileIndex));
  }
  
  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Extract quiet positions from PGN files");
  gflags::SetVersionString("1.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_input_path.empty()) {
    std::cerr << "Error: --input_path is required" << std::endl;
    gflags::ShowUsageWithFlags(argv[0]);
    return 1;
  }

  // Initialize the chess engine
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  fs::path inputPath(FLAGS_input_path);

  if (!fs::exists(inputPath)) {
    std::cerr << "Error: Path does not exist: " << inputPath << std::endl;
    return 1;
  }

  if (FLAGS_verbose) {
    std::cerr << "Processing: " << inputPath << std::endl;
    if (FLAGS_max_games > 0) {
      std::cerr << "Max games: " << FLAGS_max_games << std::endl;
    }
  }
  
  std::cerr << "Starting PGN processing (progress updates every " << PROGRESS_INTERVAL << " games)..." << std::endl;

  if (fs::is_directory(inputPath)) {
    process_directory(inputPath);
  } else if (fs::is_regular_file(inputPath)) {
    // Allow processing a single file as well
    process_pgn_file(inputPath);
  } else {
    std::cerr << "Error: Path is neither a file nor a directory: " << inputPath << std::endl;
    return 1;
  }
  
  std::cerr << "Finished processing. Total games processed: " << g_totalGamesProcessed.load() << std::endl;

  return 0;
}
