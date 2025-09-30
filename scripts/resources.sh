#!/bin/bash

# Show available system CPU RAM
echo "System Memory:"
free -h

echo
echo "Memory details (MB and GB):"
echo "=========================="
# Convert kB to MB and GB
total_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
available_kb=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
free_kb=$(grep MemFree /proc/meminfo | awk '{print $2}')
swap_total_kb=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
swap_free_kb=$(grep SwapFree /proc/meminfo | awk '{print $2}')

# Convert to MB and GB
total_mb=$((total_kb / 1024))
total_gb=$((total_mb / 1024))
available_mb=$((available_kb / 1024))
available_gb=$((available_mb / 1024))
free_mb=$((free_kb / 1024))
free_gb=$((free_mb / 1024))
swap_total_mb=$((swap_total_kb / 1024))
swap_total_gb=$((swap_total_mb / 1024))
swap_free_mb=$((swap_free_kb / 1024))
swap_free_gb=$((swap_free_mb / 1024))

echo "Total RAM:     ${total_mb} MB (${total_gb} GB)"
echo "Available RAM: ${available_mb} MB (${available_gb} GB)"
echo "Free RAM:      ${free_mb} MB (${free_gb} GB)"
echo "Total Swap:    ${swap_total_mb} MB (${swap_total_gb} GB)"
echo "Free Swap:     ${swap_free_mb} MB (${swap_free_gb} GB)"

echo
echo "Disk usage for $HOME:"
df -h "$HOME"

# Show free space in your home directory (quota-aware, if applicable)
if command -v quota &> /dev/null; then
    echo
    echo "User quota for $(whoami):"
    quota -s
fi
