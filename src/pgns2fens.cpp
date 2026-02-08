/**
 * pgns2fens - Extract quiet positions from PGN files
 *
 * Recursively traverses a directory of PGN files and outputs FEN strings
 * for "quiet" positions. A position is quiet if the next move played
 * is not a check, capture, or promotion.
 *
 * Usage: pgns2fens --input_path=<directory>
 *
 * Output: One FEN string per line to stdout.
 * Errors/warnings are written to stderr.
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include <algorithm>
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
DEFINE_bool(include_best_move, false, "Include best move comments in output");

namespace fs = std::filesystem;
using namespace ChessEngine;

// Global counter for tracking processed games
static int g_totalGamesProcessed = 0;

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
    if (end < comment.size() && (comment[end] == '+' || comment[end] == '-')) {
      end++;
    }
    while (end < comment.size() && std::isdigit(comment[end])) {
      end++;
    }
    if (end > start + 1) {
      return comment.substr(start, end - start);
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
 */
void process_game(const std::string& movetext, const std::string& startFen, const std::string& filename) {
  Position pos = startFen.empty() ? Position::init() : Position(startFen);
  std::vector<PgnToken> tokens = tokenize_movetext(movetext);

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
    if (is_quiet_move(pos, move, extMove)) {
      std::cout << pos.fen();
      if (FLAGS_include_eval && !token.eval.empty()) {
        std::cout << "|" << token.eval;
      }
      if (FLAGS_include_best_move) {
        std::cout << "|" << move.uci();
      }
      std::cout << std::endl;
    }

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
 * Processes a PGN file (possibly gzip-compressed).
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
      std::cerr << "Error: Could not open " << filepath << std::endl;
      return;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    content = ss.str();
  }

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
        if (!movetext.empty()) {
          if (FLAGS_max_games == -1 || g_totalGamesProcessed < FLAGS_max_games) {
            process_game(movetext, startFen, filepath.string());
            gameCount++;
            g_totalGamesProcessed++;
          }
          movetext.clear();
        }
        // New game starting - clear FEN from previous game
        startFen.clear();
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
      if (inMovetext && !movetext.empty()) {
        // End of game
        if (FLAGS_max_games == -1 || g_totalGamesProcessed < FLAGS_max_games) {
          process_game(movetext, startFen, filepath.string());
          gameCount++;
          g_totalGamesProcessed++;
        }
        movetext.clear();
        startFen.clear();
        inMovetext = false;
      }
      continue;
    }

    // We're in movetext
    inMovetext = true;
    movetext += " " + line;
  }

  // Process any remaining game
  if (!movetext.empty()) {
    if (FLAGS_max_games == -1 || g_totalGamesProcessed < FLAGS_max_games) {
      process_game(movetext, startFen, filepath.string());
      gameCount++;
      g_totalGamesProcessed++;
    }
  }

  if (FLAGS_verbose) {
    std::cerr << "Processed " << gameCount << " games from " << filepath << std::endl;
  }
}

/**
 * Recursively finds and processes all PGN files in a directory.
 */
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

void process_directory(const fs::path& dirpath) {
  for (const auto& entry : fs::recursive_directory_iterator(dirpath)) {
    if (entry.is_regular_file() && is_pgn_file(entry.path())) {
      if (FLAGS_max_games != -1 && g_totalGamesProcessed >= FLAGS_max_games) {
        if (FLAGS_verbose) {
          std::cerr << "Reached maximum games limit (" << FLAGS_max_games << ")" << std::endl;
        }
        break;
      }
      if (FLAGS_verbose) {
        std::cerr << "Processing file: " << entry.path() << std::endl;
      }
      process_pgn_file(entry.path());
    }
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

  if (fs::is_directory(inputPath)) {
    process_directory(inputPath);
  } else if (fs::is_regular_file(inputPath)) {
    // Allow processing a single file as well
    process_pgn_file(inputPath);
  } else {
    std::cerr << "Error: Path is neither a file nor a directory: " << inputPath << std::endl;
    return 1;
  }

  return 0;
}
