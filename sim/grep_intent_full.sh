#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "===== full enemy-encoding block in getObservation (the values version) ====="
awk '/std::vector<int> NNInterface::getObservation\(|std::array<int.*getObservation\(.*GameContext/,/return ret;/' bindings/bindings-util.cpp | sed -n '1,200p'
