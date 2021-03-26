# SIR-GN
SIR-GN: A Fast Structural Iterative Representation Learning Approach For Graph Nodes

## Folders
  * code: SIR-GN code
  * datasets: some example datasets

## Execution
Run the following command from the root:
  'python code/main.py --input_path SOURCE_GRAPH --output_path DESTINATION --embedding_size SIZE --max_it ITERATIONS'

### Arguments
  * --input_path  Input graph path
  * --output_path  Embedding size
  * --embedding_size  Maximum amount of iterations
  * --max_it  Output embedding file path

### Example
'python code/main.py --input_path datasets/karate_mirrored.txt --output_path ./sirgn_karate_mirrored_dim_20.txt  --embedding_size 20 --max_it 20'

## Datasets
Graph edge list.

### Format
  * space-separated files
  
#### Contact
Mail to Mikel Joaristi, [mikeljoaristi@u.boisestate.edu](mailto:mikeljoaristi@u.boisestate.edu) or Edoardo Serra, [edoardoserra@boisestate.edu](mailto:edoardoserra@boisestate.edu).
