#!/bin/bash
# AscensionAI VM Quick Reference
# Print this anytime:  ./vm/quickstart.sh

cat << 'GUIDE'
╔══════════════════════════════════════════════════════════════════╗
║                AscensionAI GCP VM Quick Reference                ║
╚══════════════════════════════════════════════════════════════════╝

── FIRST-TIME SETUP (do once) ────────────────────────────────────

  1. Create a SPOT VM (c3-standard-22 = fast modern cores; spot = ~3x
     cheaper). c3/c2 have real dedicated cores; AVOID e2 — its Broadwell
     cores run STS ~2-3x slower.

     gcloud compute instances create ascension-vm \
       --zone=us-west1-a \
       --machine-type=c3-standard-22 \
       --provisioning-model=SPOT \
       --instance-termination-action=STOP \
       --image-family=ubuntu-2204-lts \
       --image-project=ubuntu-os-cloud \
       --boot-disk-size=30GB

  2. SSH into the VM:
     gcloud compute ssh ascension-vm --zone=us-west1-a

  3. Push the repo's vm/ + scripts/ + external/ first (from local), then
     run the one-shot installer ON the VM:
     bash vm/install.sh
     # Installs Java 8 (NOT 17!), Xvfb, OpenAL, venv+torch, dir layout.

  4. Copy STS game files to the VM (from local PowerShell):
     gcloud compute scp --recurse ^
       "C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire\*" ^
       ascension-vm:~/ascension/game/ --zone=us-west1-a

  5. Push your code + model + seeds (from local):
     .\vm\sync.ps1 push

── DAILY WORKFLOW ─────────────────────────────────────────────────

  Make code changes locally, then:

  .\vm\sync.ps1 push-code           # send code changes
  gcloud compute ssh ascension-vm    # SSH in
  cd ~/ascension
  ./vm/stop.sh                       # stop if running
  ./vm/run_training.sh --workers 8 --hours 12

  Later (from local):
  .\vm\sync.ps1 pull                 # get model + logs back
  .\vm\sync.ps1 status               # or just check progress

── TRAINING COMMANDS (on the VM) ──────────────────────────────────

  # 8 workers, 12 hours (good default for c3-standard-22 / 22 vCPU)
  ./vm/run_training.sh --workers 8 --hours 12

  # 24-hour overnight run
  ./vm/run_training.sh --workers 8 --hours 24

  # Workers only, no trainer (just collecting rollouts)
  ./vm/run_training.sh --workers 8 --hours 8 --no-trainer

  # Stop everything
  ./vm/stop.sh

  Sizing rule of thumb: each STS worker eats ~2 vCPUs (game) + ~0.7 (python).
  On 22 vCPUs, 8 workers is the sweet spot. More just adds contention.

── EVAL COMMANDS (on the VM) ──────────────────────────────────────

  ./vm/run_eval.sh --games 200
  ./vm/run_eval.sh --games 200 --policy heuristic --tag heuristic_200
  ./vm/run_eval.sh --games 200 --instances 4      # parallel, faster

── SYNC COMMANDS (from local PowerShell) ──────────────────────────

  .\vm\sync.ps1 push          # Push everything to VM
  .\vm\sync.ps1 pull          # Pull model + logs back
  .\vm\sync.ps1 push-code     # Push only code (after edits)
  .\vm\sync.ps1 pull-model    # Pull only the .pt file
  .\vm\sync.ps1 push-model    # Push only the .pt file
  .\vm\sync.ps1 status        # Check VM training remotely

── SPOT PREEMPTION & RECOVERY ─────────────────────────────────────

  Spot VMs can be reclaimed by GCP at any time. With
  --instance-termination-action=STOP the VM STOPS (disk preserved),
  it is not deleted. To recover:

     gcloud compute instances start ascension-vm --zone=us-west1-a
     gcloud compute ssh ascension-vm --zone=us-west1-a
     cd ~/ascension && ./vm/run_training.sh --workers 8 --hours 12

  The model checkpoint + training_stats.csv survive on disk, so training
  resumes from where it left off.

── CHANGING MACHINE TYPE / CONVERTING TO SPOT ─────────────────────

  You CANNOT flip an existing standard VM to spot in place — you must
  delete + recreate the instance while preserving the boot disk:

     # 1. Keep the disk when the instance is deleted
     gcloud compute instances set-disk-auto-delete ascension-vm \
       --zone=us-west1-a --no-auto-delete --disk=ascension-vm
     # 2. Stop + delete the instance (disk stays)
     gcloud compute instances stop   ascension-vm --zone=us-west1-a
     gcloud compute instances delete ascension-vm --zone=us-west1-a --quiet
     # 3. Recreate as spot, re-attaching the SAME disk
     gcloud compute instances create ascension-vm \
       --zone=us-west1-a \
       --machine-type=c3-standard-22 \
       --provisioning-model=SPOT \
       --instance-termination-action=STOP \
       --disk=name=ascension-vm,boot=yes,auto-delete=yes

── MONITORING (on the VM, use tmux) ───────────────────────────────

  tmux new -s training
  ./vm/run_training.sh --workers 8 --hours 12
  # Ctrl+B then D to detach (training keeps running)
  # tmux attach -t training   to reattach

  tail -f ~/ascension/logs/trainer.log
  tail -f ~/ascension/logs/worker_1.log

── COST TRACKING (spot, us-west1) ─────────────────────────────────

  c3-standard-22 spot:  ~$0.19/hr  = ~$4.56/day
  e2-standard-16 spot:  ~$0.16/hr  (cheaper, but slow cores — not worth it)

  Free credits: $263.49, expires Aug 7 2026.
  c3-standard-22 spot ~24/7 ≈ $137/mo — fine for bursts, watch the timer.

GUIDE
