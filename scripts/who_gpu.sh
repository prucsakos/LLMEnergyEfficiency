nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid \
  --format=csv,noheader,nounits | \
while IFS=, read -r pid pname used uuid; do
  user=$(ps -o user= -p "$pid")
  printf "%-12s PID=%-8s mem=%.2f GiB  GPU=%s  cmd=%s\n" "$user" "$pid" "$(awk "BEGIN{print $used/1024}")" "$uuid" "$pname"
done