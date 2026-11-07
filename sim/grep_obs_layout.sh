#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "== intent/move mentions in bindings =="
grep -in "intent\|moveHistory\|getMove" bindings/bindings-util.cpp | head -20
echo "== combat observation block (battle section) =="
grep -n "battle\|monster\|player" bindings/bindings-util.cpp | grep -in "observ\|encode\|offset" | head -20
echo "== the getObservation function body =="
awk '/getObservationMaximums|std::vector<int> getObservation|void getObservation/,0' bindings/bindings-util.cpp | head -120
