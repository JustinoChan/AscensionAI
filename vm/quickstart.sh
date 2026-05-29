#!/bin/bash
# AscensionAI VM Quick Reference
# Print this anytime:  ./vm/quickstart.sh

cat << 'GUIDE'
╔══════════════════════════════════════════════════════════════════╗
║                AscensionAI GCP VM Quick Reference               ║
╚══════════════════════════════════════════════════════════════════╝

── FIRST-TIME SETUP (do once) ────────────────────────────────────

  1. Create a GCP VM:
     gcloud compute instances create ascension-vm \
       --zone=us-west1-a \
       --machine-type=e2-standard-8 \
       --provisioning-model=SPOT \
       --image-family=ubuntu-2204-lts \
       --image-project=ubuntu-os-cloud \
       --boot-disk-size=30GB

  2. SSH into the VM:
     gcloud compute ssh ascension-vm --zone=us-west1-a

  3. Run the setup script on the VM:
     bash vm/setup.sh

  4. Copy STS game files to the VM (from local PowerShell):
     gcloud compute scp --recurse ^
       "C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire" ^
       ascension-vm:~/ascension/game/ --zone=us-west1-a

  5. Push your code + model (from local):
     .\vm\sync.ps1 push

── DAILY WORKFLOW ─────────────────────────────────────────────────

  Make code changes locally, then:

  .\vm\sync.ps1 push-code           # send code changes
  gcloud compute ssh ascension-vm    # SSH in
  cd ~/ascension
  ./vm/stop.sh                       # stop if running
  ./vm/run_training.sh --workers 8 --hours 12 --restart-every 50

  Later (from local):
  .\vm\sync.ps1 pull                 # get model + logs back
  .\vm\sync.ps1 status               # or just check progress

── TRAINING COMMANDS (on the VM) ──────────────────────────────────

  # 8 workers, 12 hours, restart every 50 games
  ./vm/run_training.sh --workers 8 --hours 12 --restart-every 50

  # 6 workers, 24-hour overnight run
  ./vm/run_training.sh --workers 6 --hours 24 --restart-every 40

  # Workers only, no trainer (collecting rollouts)
  ./vm/run_training.sh --workers 10 --hours 8 --no-trainer

  # Stop everything
  ./vm/stop.sh

── EVAL COMMANDS (on the VM) ──────────────────────────────────────

  # Standard 200-game eval
  ./vm/run_eval.sh --games 200

  # Heuristic baseline
  ./vm/run_eval.sh --games 200 --policy heuristic --tag heuristic_200

  # Parallel eval (4 instances, faster)
  ./vm/run_eval.sh --games 200 --instances 4

── SYNC COMMANDS (from local PowerShell) ──────────────────────────

  .\vm\sync.ps1 push          # Push everything to VM
  .\vm\sync.ps1 pull          # Pull model + logs back
  .\vm\sync.ps1 push-code     # Push only code (after edits)
  .\vm\sync.ps1 pull-model    # Pull only the .pt file
  .\vm\sync.ps1 push-model    # Push only the .pt file
  .\vm\sync.ps1 status        # Check VM training remotely

── MOVING .pt FILES ───────────────────────────────────────────────

  # Local → VM (PowerShell):
  gcloud compute scp models\ppo_sts.pt ^
    ascension-vm:~/ascension/models/ --zone=us-west1-a

  # VM → Local (PowerShell):
  gcloud compute scp ^
    ascension-vm:~/ascension/models/ppo_sts.pt ^
    models\ --zone=us-west1-a

  # Or use the sync script:
  .\vm\sync.ps1 push-model
  .\vm\sync.ps1 pull-model

── MONITORING (on the VM, use tmux) ───────────────────────────────

  tmux new -s training
  ./vm/run_training.sh --workers 8 --hours 12
  # Ctrl+B then D to detach (training keeps running)
  # tmux attach -t training   to reattach

  # Watch logs:
  tail -f ~/ascension/logs/trainer.log
  tail -f ~/ascension/logs/worker_1_debug.log

── COST TRACKING ──────────────────────────────────────────────────

  e2-standard-8 spot:  ~$0.07/hr  = ~$1.70/day
  e2-standard-16 spot: ~$0.16/hr  = ~$3.84/day

  Your free credits: $263.49, expires Aug 7 2026
  At e2-standard-8 spot:  ~$1.70/day × 70 days = ~$119 total
  Leaves ~$144 buffer for bigger burst runs or evals.

GUIDE
