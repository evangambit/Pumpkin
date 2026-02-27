from huggingface_hub import snapshot_download

repo_id = "official-stockfish/fishtest_pgns"
repo_type = "dataset"
# allow_pattern = "18-0[56789]*/*/*.pgn.gz"
# allow_pattern = "18-1[01234]-*/*/*.pgn.gz"
local_dir = "./pgns"

snapshot_download(repo_id=repo_id, repo_type=repo_type, allow_patterns=allow_pattern, local_dir=local_dir)
