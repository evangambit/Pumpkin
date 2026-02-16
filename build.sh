
#!/bin/bash

# Base compiler flags
CXXFLAGS="-std=c++20"
LDFLAGS="-pthread -L/usr/local/lib -lgflags -lz"
INCLUDE_FLAGS=""

# Detect operating system and add platform-specific flags
# Add optimization flags for all platforms
CXXFLAGS="$CXXFLAGS -O3 -DNDEBUG -march=native"

if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS - check for Homebrew paths
  if [[ -d "/opt/homebrew" ]]; then
    # Apple Silicon Mac with Homebrew
    INCLUDE_FLAGS="-I/opt/homebrew/include"
    LDFLAGS="$LDFLAGS -L/opt/homebrew/lib"
  elif [[ -d "/usr/local/include" ]]; then
    # Intel Mac with Homebrew
    INCLUDE_FLAGS="-I/usr/local/include"
  fi
fi

# Build command
g++ $CXXFLAGS -o $1 $2 $(find src/ -name "*.cpp" | grep -Ev "(Tests|[Mm]ain|make_tables|make_moveorder_tables)\\.cpp") $INCLUDE_FLAGS $LDFLAGS model_data.cpp qst_data.cpp ${@:3}
