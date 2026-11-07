#!/bin/bash
pkill -f '[t]rain_search'
pkill -f '[s]im_worker_search'
sleep 2
echo "trainers: $(pgrep -fc '[t]rain_search' || echo 0)"
echo "workers:  $(pgrep -fc '[s]im_worker_search' || echo 0)"
