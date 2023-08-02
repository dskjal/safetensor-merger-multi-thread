# Safetensor Merger Multi Thread
- Merge with CPU
- Less consume RAM since using a memory-mapped file
- Multi-threading merges fast

# How to use
### command example
python .\merger.py Model_A_path Model_B_path Output_model_path Merge_ratio Memory_type --num_thread 12 --weight_file_path weights.txt

### Command Arguments
|Command|Description|
|:---:|:---|
|pathA|safetensorA path|
|pathB|safetensorB path|
|out|output safetensor path|
|ratio|Merge ratio. ratio * A + (1-ratio) * B. --weight_file_path takes precedence|
|dtype|Weight type. Default is F16. Following type is usable. F64, F32, F16, I64, I32, I16, I8, U8, BOOL.|
|--weight_file_path|Custom Block Weight. See [Merge Block Weighted - GUI](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui)|
|--num_thread|Number of threads to use. Use os.cpu_count() if default. Use lesser RAM if num_thread is 1|
|--meta_data|Metadata|
|--allow_overwrite|Overwrite output safetensor file|
|--discard_metadata|Don't write metadata|

# Benchmark
### Environment
- Ryzen 5 2600 (6 core 12 thread)
- RAM 32 GB
- file size 6.7 GB SDXL
- NVMe SSD

### Result
|Thread|RAM Usage(GB)|Time(sec)|
|:---:|---:|---:|
|1|8|180|
|6|15|26|
|12|15|22|

# 概要
- CPU でマージ
- メモリマップトファイルを使用するので省メモリ
- マルチスレッド動作

# 使い方
### コマンド例
python .\merger.py Model_A_path Model_B_path Output_model_path Merge_ratio Memory_type --num_thread 12 --weight_file_path weights.txt

### 引数リスト
|Command|Description|
|:---:|:---|
|pathA|safetensorA パス|
|pathB|safetensorB パス|
|out|出力 safetensor パス|
|ratio|マージ率. ratio * A + (1-ratio) * B. --weight_file_path がある場合はそちらが優先される|
|dtype|ウェイトタイプ. F16 がデフォルト. F64, F32, F16, I64, I32, I16, I8, U8, BOOL が有効|
|--weight_file_path|階層マージのウェイト. [Merge Block Weighted - GUI](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui) を参照|
|--num_thread|スレッド数。デフォルトでは os.cpu_count() が使われる。１を指定すると省メモリ動作になる|
|--meta_data|メタデータ|
|--allow_overwrite|出力ファイルを上書きする|
|--discard_metadata|メタデータを出力しない|

# ベンチ
### 環境
- Ryzen 5 2600 (6 core 12 thread)
- RAM 32 GB
- file size 6.7 GB SDXL
- NVMe SSD

### 結果
|Thread|RAM Usage(GB)|Time(sec)|
|:---:|---:|---:|
|1|8|180|
|6|15|26|
|12|15|22|
