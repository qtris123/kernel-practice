#!/bin/bash

# Usage: ./wsl_run.sh -a [ring|bde] -n [num_nodes] -t [tensor_size] -p [port]

algo="ring"
num_nodes=4
tensor_size=1024
port=29500

while getopts "a:n:t:p:h" opt; do
  case $opt in
    a) algo="$OPTARG" ;;
    n) num_nodes="$OPTARG" ;;
    t) tensor_size="$OPTARG" ;;
    p) port="$OPTARG" ;;
    h)
      echo "Usage: $0 -a [ring|bde] -n [num_nodes] -t [tensor_size] -p [port]"
      exit 0
      ;;
    *)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
  esac
done

if [ "$algo" = "ring" ]; then
  script="ring_allreduce.py"
elif [ "$algo" = "bde" ]; then
  script="recursive_hd_reduce.py"
else
  echo "Unknown algorithm: $algo (choose 'ring' or 'bde')"
  exit 1
fi

export USE_REAL_DIST=1

echo "=========================================================="
echo "Algorithm:   $algo ($script)"
echo "Workers:     $num_nodes CPU processes"
echo "Tensor Size: $tensor_size elements ($(( tensor_size * 4 )) bytes)"
echo "Backend:     PyTorch Gloo over TCP (127.0.0.1:$port)"
echo "=========================================================="

start_time=$(date +%s%N)

for (( rank=0; rank<num_nodes; rank++ )); do
  python3 "$script" --master-ip "127.0.0.1:$port" --num-nodes "$num_nodes" --rank "$rank" --tensor-size "$tensor_size" &
done

wait

end_time=$(date +%s%N)
elapsed_ms=$(( (end_time - start_time) / 1000000 ))

echo "=========================================================="
echo "All $num_nodes ranks finished successfully in ${elapsed_ms} ms"
echo "=========================================================="
