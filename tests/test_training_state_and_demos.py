from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ppo_model import PPOTrainer
from train_offline import _bc_demo_candidates, load_bc_demo


class TrainingStateAndDemoTests(unittest.TestCase):
    def test_ppo_checkpoint_preserves_auto_tuned_lr_and_entropy(self):
        tmp_root = ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            checkpoint = str(Path(tmp) / "ppo.pt")
            trainer = PPOTrainer(obs_size=4, n_actions=3, lr=0.01, ent_coef=0.007)
            trainer.set_lr(0.0025)
            trainer.bc_coef = 0.033
            trainer.total_updates = 12
            trainer.save(checkpoint)

            loaded = PPOTrainer(obs_size=4, n_actions=3, lr=0.1, ent_coef=0.001)
            loaded.load(checkpoint, load_hparams=True)

            self.assertEqual(12, loaded.total_updates)
            self.assertAlmostEqual(0.0025, loaded.get_lr())
            self.assertAlmostEqual(0.007, loaded.ent_coef)
            self.assertAlmostEqual(0.033, loaded.bc_coef)
            self.assertAlmostEqual(0.033, loaded._loaded_bc_coef)

    def test_hparam_load_does_not_override_explicit_structural_args(self):
        tmp_root = ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            checkpoint = str(Path(tmp) / "ppo.pt")
            trainer = PPOTrainer(
                obs_size=4, n_actions=3, lr=0.01, ent_coef=0.007,
                clip_range=0.1, target_kl=0.01, n_epochs=2, batch_size=8,
            )
            trainer.save(checkpoint)

            loaded = PPOTrainer(
                obs_size=4, n_actions=3, lr=0.1, ent_coef=0.001,
                clip_range=0.2, target_kl=0.03, n_epochs=4, batch_size=64,
            )
            loaded.load(checkpoint, load_hparams=True)

            self.assertEqual(4, loaded.n_epochs)
            self.assertEqual(64, loaded.batch_size)
            self.assertAlmostEqual(0.2, loaded.clip_range)
            self.assertAlmostEqual(0.03, loaded.target_kl)

    def test_plain_checkpoint_load_keeps_constructor_entropy(self):
        tmp_root = ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            checkpoint = str(Path(tmp) / "ppo.pt")
            trainer = PPOTrainer(obs_size=4, n_actions=3, lr=0.01, ent_coef=0.007)
            trainer.save(checkpoint)

            loaded = PPOTrainer(obs_size=4, n_actions=3, lr=0.1, ent_coef=0.001)
            loaded.load(checkpoint)

            self.assertAlmostEqual(0.001, loaded.ent_coef)

    def test_load_bc_demo_merges_directory_sources(self):
        tmp_root = ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            demo_dir = Path(tmp)
            for i in range(2):
                np.savez_compressed(
                    demo_dir / f"demo_{i}.npz",
                    observations=np.full((2, 4), i, dtype=np.float32),
                    actions=np.array([i, i + 1], dtype=np.int64),
                    action_masks=np.ones((2, 3), dtype=np.bool_),
                )

            obs, actions, masks = load_bc_demo(str(demo_dir))

            self.assertEqual((4, 4), obs.shape)
            self.assertEqual([0, 1, 1, 2], actions.tolist())
            self.assertEqual((4, 3), masks.shape)

    def test_default_bc_demo_candidates_include_shared_demos(self):
        model_path = str(ROOT / "models" / "ppo_sts.pt")

        candidates = _bc_demo_candidates(model_path, None)

        self.assertEqual(str(ROOT / "models" / "ppo_sts_bc_demos.npz"), candidates[0])
        self.assertIn(str(ROOT / "bc_demos_shared"), candidates)


if __name__ == "__main__":
    unittest.main()
