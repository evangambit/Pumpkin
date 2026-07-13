cmake_minimum_required(VERSION 3.16)

project(pumpkin LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PUMPKIN_BUILD_TESTS "Build unit tests" ON)
option(PUMPKIN_NATIVE_ARCH "Tune for the host CPU (-march=native)" ON)
option(PUMPKIN_LTO "Enable link-time optimization in Release builds" ON)

if(PUMPKIN_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
  else()
    message(WARNING "LTO not supported: ${ipo_error}")
  endif()
endif()

find_package(Threads REQUIRED)

add_library(pumpkin_engine STATIC
  byhand_data.cpp
  model_data.cpp
  qst_data.cpp
  src/eval/nnue/Utils.cpp
  src/eval/pst/PieceSquareEvaluator.cpp
  src/game/CreateThreats.cpp
  src/game/Geometry.cpp
  src/game/Move.cpp
  src/game/Position.cpp
  src/game/Utils.cpp
  src/game/movegen/movegen.cpp
  src/search/search.cpp
  src/search/transposition_table.cpp
  src/utils/StringUtils.cpp
)

target_include_directories(pumpkin_engine PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(pumpkin_engine PUBLIC Threads::Threads)

add_executable(uci src/uci/main.cpp)
target_link_libraries(uci PRIVATE pumpkin_engine)

set(PUMPKIN_PARAM "" CACHE STRING "Optional compile-time PARAM value")
if(PUMPKIN_PARAM)
  target_compile_definitions(uci PRIVATE PARAM=${PUMPKIN_PARAM})
endif()

set(PUMPKIN_DEFAULT_EVALUATOR "" CACHE STRING "Default evaluator at startup: n=nnue, b=byhand, empty=pst")
if(PUMPKIN_DEFAULT_EVALUATOR STREQUAL "n")
  target_compile_definitions(uci PRIVATE PUMPKIN_DEFAULT_EVALUATOR_NNUE=1)
elseif(PUMPKIN_DEFAULT_EVALUATOR STREQUAL "b")
  target_compile_definitions(uci PRIVATE PUMPKIN_DEFAULT_EVALUATOR_BYHAND=1)
elseif(PUMPKIN_DEFAULT_EVALUATOR)
  message(FATAL_ERROR "PUMPKIN_DEFAULT_EVALUATOR must be n, b, or empty (got '${PUMPKIN_DEFAULT_EVALUATOR}')")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(_pumpkin_release_opts -funroll-loops)
  if(PUMPKIN_NATIVE_ARCH)
    list(APPEND _pumpkin_release_opts -march=native)
  endif()
  if(PUMPKIN_LTO)
    list(APPEND _pumpkin_release_opts -flto)
  endif()

  # negamax and other hot paths are header-only and instantiated in uci/main.cpp,
  # so release tuning must apply to the uci target as well as pumpkin_engine.
  foreach(_target IN ITEMS pumpkin_engine uci)
    target_compile_options(${_target} PRIVATE
      $<$<CONFIG:Release>:${_pumpkin_release_opts}>
      $<$<CONFIG:RelWithDebInfo>:${_pumpkin_release_opts}>
    )
    if(PUMPKIN_LTO)
      set_property(TARGET ${_target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
      set_property(TARGET ${_target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    endif()
  endforeach()

  if(PUMPKIN_LTO)
    target_link_options(uci PRIVATE
      $<$<CONFIG:Release>:-flto>
      $<$<CONFIG:RelWithDebInfo>:-flto>
    )
  endif()
endif()

if(PUMPKIN_BUILD_TESTS)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    message(WARNING "GTest not found; disabling tests (install libgtest-dev or pass -DPUMPKIN_BUILD_TESTS=OFF)")
    set(PUMPKIN_BUILD_TESTS OFF CACHE BOOL "Build unit tests" FORCE)
  endif()
endif()

if(PUMPKIN_BUILD_TESTS)
  enable_testing()

  file(GLOB_RECURSE PUMPKIN_TEST_SOURCES CONFIGURE_DEPENDS
    "src/**/tests/*Tests.cpp"
  )

  add_executable(test_runner ${PUMPKIN_TEST_SOURCES})
  target_link_libraries(test_runner PRIVATE
    pumpkin_engine
    GTest::gtest
    GTest::gtest_main
  )

  include(GoogleTest)
  gtest_discover_tests(test_runner)
endif()
