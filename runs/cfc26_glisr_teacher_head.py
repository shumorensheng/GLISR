"""Training-only author CNN feature teacher for the existing GLISR YOLO head."""
import torch
from torch import nn
from torch.nn import functional as F

from runs import cfc26_glisr_yolo_adaptation as parent
from runs.cfc26_glisr_yolo_head import AUTHOR_WEIGHTS


class TeacherPyramidHead(parent.YOLOTaskHead):
    """Set teacher_images to the actual RGB256 crop batch /255 before training.

    The frozen author backbone supplies training supervision only. It is outside
    the student's registered parameters/state_dict; eval uses the parent decoder
    and NMS without teacher execution. Student checkpoint keys remain unchanged.
    """
    def __init__(self, freeze_neck: bool):
        super().__init__()
        self.freeze_neck = freeze_neck
        source = torch.load(AUTHOR_WEIGHTS, map_location='cpu', weights_only=False)['model']
        teacher = nn.Sequential(*list(source.model)[:10]).float().requires_grad_(False).eval()
        object.__setattr__(self, 'teacher', teacher)
        self.teacher_batches = 0
        self.teacher_samples = 0
        self.teacher_cuda_moves = 0
        self.distillation_witness = {}
        self._teacher_initial_versions = self._teacher_versions()
        if freeze_neck:
            self.core.model.requires_grad_(False)
        self.train()

    def cuda(self, device=None):
        super().cuda(device)
        self.teacher.cuda(device)
        self.teacher_cuda_moves += 1
        self._teacher_initial_versions = self._teacher_versions()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        if self.freeze_neck:
            for module in self.core.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, field, targets=None):
        losses = super().forward(field, targets)
        if self.training:
            with torch.no_grad():
                x = self.teacher_images
                teacher_features = []
                # Actual author layers0..9 all have f=-1; taps are P3/P4/SPPF.
                for layer in self.teacher:
                    x = layer(x)
                    if layer.i in (4, 6, 9):
                        teacher_features.append(x)
            normalized_mse = torch.stack([
                F.mse_loss(student, teacher) / (teacher.square().mean() + 1e-6)
                for student, teacher in zip(self.core.pyramid_features, teacher_features)
            ]).mean()
            distillation = .25 * normalized_mse
            losses['teacher_pyramid_loss'] = distillation
            self.teacher_batches += 1
            self.teacher_samples += self.teacher_images.shape[0]
            self.distillation_witness = {
                'student_shapes': [list(x.shape) for x in self.core.pyramid_features],
                'teacher_shapes': [list(x.shape) for x in teacher_features],
                'teacher_image_shape': list(self.teacher_images.shape),
                'normalized_mse': float(normalized_mse.detach()),
                'weighted_loss': float(distillation.detach()),
                'lambda': .25,
            }
        return losses

    def _teacher_versions(self):
        return {name: tensor._version for name, tensor in
                (*self.teacher.named_parameters(), *self.teacher.named_buffers())}

    def teacher_report(self):
        versions = self._teacher_versions()
        return {
            'freeze_neck': self.freeze_neck,
            'teacher_source': str(AUTHOR_WEIGHTS),
            'teacher_source_routing': [{'i': layer.i, 'f': layer.f} for layer in self.teacher],
            'teacher_parameter_count': sum(p.numel() for p in self.teacher.parameters()),
            'teacher_trainable_parameters': sum(p.numel() for p in self.teacher.parameters() if p.requires_grad),
            'teacher_all_grad_none': all(p.grad is None for p in self.teacher.parameters()),
            'teacher_all_modules_eval': all(not m.training for m in self.teacher.modules()),
            'teacher_device': str(next(self.teacher.parameters()).device),
            'teacher_cuda_moves': self.teacher_cuda_moves,
            'teacher_batches': self.teacher_batches,
            'teacher_samples': self.teacher_samples,
            'teacher_version_witness': {
                'kind': 'PyTorch tensor _version counters, not hashes or bitwise equality',
                'baseline': 'After construction or latest explicit cuda move',
                'parameter_tensors': len(list(self.teacher.parameters())),
                'buffer_tensors': len(list(self.teacher.buffers())),
                'unchanged': versions == self._teacher_initial_versions,
                'initial_version_values': sorted(set(self._teacher_initial_versions.values())),
                'current_version_values': sorted(set(versions.values())),
                'changed': {name: [self._teacher_initial_versions[name], version]
                            for name, version in versions.items()
                            if version != self._teacher_initial_versions[name]},
            },
            'neck_trainable_parameters': sum(p.numel() for p in self.core.model.parameters() if p.requires_grad),
            'neck_batchnorm_all_eval': all(not m.training for m in self.core.model.modules()
                                           if isinstance(m, nn.BatchNorm2d)),
            'detect_training': self.core.model[-1].training,
            'distillation': self.distillation_witness,
        }
