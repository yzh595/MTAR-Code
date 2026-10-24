"""CPU smoke/regression checks; no ImageNet or pretrained checkpoint required."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from autoregressive.models.gpt import ModelArgs as DecoderArgs, Transformer as Decoder
from autoregressive.models.generate import generate
from mtarpp.checkpoint import load_ntp_weights
from mtarpp.dataset import CustomDataset
from mtarpp.model import ModelArgs, Transformer
from mtarpp.teacher import ModelArgs as TeacherArgs, Transformer as Teacher
from mtarpp.train import (
    build_dino_keep_indices, build_grouped_teacher_indices,
    build_sparse_proximity_mask, get_epoch_drop_rate,
)


def tiny_model():
    return Transformer(ModelArgs(
        dim=64, n_layer=2, n_head=4, multiple_of=16, vocab_size=32,
        num_classes=4, block_size=16, student_target_layers=[2],
        teacher_dim=32, projector_dim=64, future_feature_pred_layer=2,
        future_feature_offsets=[1, 4], loss_weight_contrast=0.2,
        contrast_num_samples=8, token_dropout_p=0, resid_dropout_p=0,
        ffn_dropout_p=0,
    ))


class MtarppTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def test_schedule(self):
        self.assertEqual(get_epoch_drop_rate(239, 300), 0.5)
        self.assertEqual(get_epoch_drop_rate(240, 300), 0.0)
        self.assertEqual(get_epoch_drop_rate(299, 300), 0.0)

    def test_power_probabilities_and_unique_sorted_ids(self):
        scores = torch.tensor([[0., 0.1, 0.2, 0.4, 0.8, 1., 2., 4.]])
        original = torch.multinomial
        with patch('torch.multinomial', wraps=original) as sample:
            ids = build_dino_keep_indices(scores, 1, 8, 0.5, 'cpu', 0.35)
        probabilities = sample.call_args.args[0]
        expected = scores.clamp_min(1e-9).pow(1 / 0.35)
        expected /= expected.sum(dim=1, keepdim=True)
        torch.testing.assert_close(probabilities, expected)
        self.assertFalse(sample.call_args.kwargs['replacement'])
        self.assertEqual(ids.shape, (1, 4))
        self.assertTrue((ids[:, 1:] > ids[:, :-1]).all())

    def test_no_drop_keeps_all_original_positions(self):
        scores = torch.rand(2, 255)
        ids = build_dino_keep_indices(scores, 2, 255, 0, 'cpu', 0.35)
        torch.testing.assert_close(ids, torch.arange(255).expand(2, -1))

    def test_teacher_groups_keep_all_targets_with_fixed_budget(self):
        scores = torch.rand(3, 255)
        ids, anchors, counts, covered = build_grouped_teacher_indices(
            scores, 3, 255, 0.5, [1, 16], 'cpu',
        )
        self.assertEqual(ids.shape, (3, 128))
        retained = torch.zeros(3, 255, dtype=torch.bool).scatter_(1, ids, True)
        for row in range(3):
            positions = anchors[row].nonzero().flatten()
            self.assertEqual(len(positions), counts[row].item())
            self.assertTrue((positions + 16 < 255).all())
            for offset in (0, 1, 16):
                self.assertTrue(retained[row, positions + offset].all())
            self.assertLessEqual(covered[row].item(), 128)

    def test_sparse_teacher_mask_uses_original_positions(self):
        full = torch.arange(2 * 17 * 17).reshape(2, 17, 17)
        ids = torch.tensor([[0, 3, 10, 15], [2, 4, 12, 15]])
        cropped = build_sparse_proximity_mask(
            SimpleNamespace(proximity_mask=full), ids, 1, 17, 'cpu',
        )
        for row in range(2):
            positions = torch.cat((torch.tensor([0]), ids[row] + 1))
            torch.testing.assert_close(cropped[row], full[row][positions][:, positions])

    def test_sparse_student_forward_original_ce_and_backward(self):
        model = tiny_model().train()
        targets = torch.randint(0, 32, (2, 16))
        ids = torch.tensor([[0, 2, 4, 6, 8, 10, 12, 14]]).expand(2, -1)
        logits, features, ce, kept, predictions, tcr = model(
            idx=targets, cond_idx=torch.tensor([0, 1]), targets=targets,
            ids_keep=ids, physical_sparse_input=True,
        )
        original_targets = torch.gather(
            targets, 1, torch.cat((torch.zeros(2, 1, dtype=torch.long), ids + 1), 1),
        )
        expected = F.cross_entropy(logits.flatten(0, 1), original_targets.flatten())
        torch.testing.assert_close(ce, expected)
        torch.testing.assert_close(kept, ids)
        self.assertEqual(features[0].shape, (2, 8, 32))
        self.assertEqual(len(predictions), 2)
        loss = ce + sum(value.square().mean() for value in features + predictions) + .2 * tcr
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertIsNotNone(model.future_feature_heads[0].fc1.weight.grad)

    def test_standard_ntp_checkpoint_and_cached_decoding_parity(self):
        student = tiny_model().eval()
        with torch.no_grad():
            student.output.weight.normal_(std=0.02)
        decoder = Decoder(DecoderArgs(
            dim=64, n_layer=2, n_head=4, multiple_of=16, vocab_size=32,
            num_classes=4, block_size=16,
        )).eval()
        load_ntp_weights(decoder, student.state_dict())
        labels = torch.tensor([0, 1])
        with torch.no_grad():
            # Both eval implementations use prefill + cached single-token decode,
            # not the training-mode dense-input call contract.
            actual = generate(student, labels, 16, cfg_scale=1.8,
                              temperature=1.0, top_k=0, top_p=1.0,
                              sample_logits=False)
            generated = generate(decoder, labels, 16, cfg_scale=1.8,
                                 temperature=1.0, top_k=0, top_p=1.0,
                                 sample_logits=False)
        torch.testing.assert_close(actual, generated)
        self.assertEqual(generated.shape, (2, 16))
        self.assertTrue(((generated >= 0) & (generated < 32)).all())

    def test_checkpoint_rejects_missing_backbone(self):
        student = tiny_model()
        state = dict(student.state_dict())
        state.pop('tok_embeddings.weight')
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            load_ntp_weights(student, state)

    def test_teacher_sparse_forward_physical_sequence_length(self):
        teacher = Teacher(TeacherArgs(
            dim=64, n_layer=2, n_head=4, multiple_of=16, vocab_size=32,
            num_classes=4, block_size=16,
        )).eval()
        lengths = []
        hook = teacher.layers[1].register_forward_hook(
            lambda module, inputs, output: lengths.append(output.shape[1]),
        )
        with torch.no_grad():
            teacher(
                idx=torch.randint(0, 32, (2, 16)), cond_idx=torch.tensor([0, 1]),
                ids_keep=torch.tensor([[0, 2, 4, 7, 15]]).expand(2, -1),
                mask=torch.ones(2, 6, 6, dtype=torch.bool),
            )
        hook.remove()
        self.assertEqual(lengths, [6])

    def test_dataset_aligned_tencrop_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind in ('codes', 'labels', 'scores'):
                (root / f'imagenet256_{kind}').mkdir()
            views = np.broadcast_to(np.arange(10)[None, :, None], (1, 10, 256)).copy()
            np.save(root / 'imagenet256_codes/0.npy', views.astype(np.int16))
            np.save(root / 'imagenet256_scores/0.npy', views.astype(np.float32))
            np.save(root / 'imagenet256_labels/0.npy', np.array([3]))
            dataset = CustomDataset(str(root / 'imagenet256_codes'), str(root / 'imagenet256_labels'))
            codes, labels, scores = dataset[0]
            torch.testing.assert_close(codes.float(), scores)
            self.assertEqual(codes.shape, (1, 256))
            self.assertEqual(labels.item(), 3)

    def test_dataset_requires_matching_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind in ('codes', 'labels', 'scores'):
                (root / f'imagenet256_{kind}').mkdir()
            np.save(root / 'imagenet256_codes/0.npy', np.zeros((1, 2, 256)))
            with self.assertRaisesRegex(ValueError, 'matching'):
                CustomDataset(str(root / 'imagenet256_codes'), str(root / 'imagenet256_labels'))

    def test_recipe_matches_launch_constants(self):
        root = Path(__file__).resolve().parents[1]
        recipe = json.loads((root / 'configs/mtarpp/imagenet256_gptl_300ep.json').read_text())
        launcher = (root / 'train_c2i_mtarpp.sh').read_text()
        self.assertEqual(recipe['student_sampling'], 'power_normalized_without_replacement')
        self.assertEqual(recipe['feature_offsets'], [0, 1, 16])
        for flag in ('--epochs 300', '--lr 1e-4', '--global-batch-size 256',
                     '--student-target-layers 24', '--teacher-target-layers 36'):
            self.assertIn(flag, launcher)

    def test_launchers_dry_run_and_paths_with_spaces(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='mtarpp dry run ') as temporary:
            work = Path(temporary)
            data = work / 'features with spaces'
            for kind in ('codes', 'labels', 'scores'):
                (data / f'imagenet256_{kind}').mkdir(parents=True)
            checkpoint = work / 'placeholder checkpoint.pt'
            checkpoint.write_bytes(b'dry-run-only')
            environment = dict(os.environ, DRY_RUN='1', CUDA_VISIBLE_DEVICES='0,1')
            train = subprocess.run(
                ['bash', str(root / 'train_c2i_mtarpp.sh'), str(data),
                 str(checkpoint), str(work / 'output with spaces')],
                cwd=work, env=environment, capture_output=True, text=True, check=True,
            )
            self.assertIn('--nproc_per_node=2', train.stdout)
            self.assertIn('mtarpp.train', train.stdout)
            sample = subprocess.run(
                ['bash', str(root / 'sample_c2i_mtarpp.sh'), str(checkpoint),
                 str(checkpoint), '1.80', str(work / 'samples with spaces'), '50000'],
                cwd=work, env=environment, capture_output=True, text=True, check=True,
            )
            self.assertIn('mtarpp.sample', sample.stdout)
            self.assertIn('--num-fid-samples 50000', sample.stdout)


if __name__ == '__main__':
    unittest.main()
