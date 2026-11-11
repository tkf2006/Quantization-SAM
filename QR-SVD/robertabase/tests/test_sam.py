import torch
import torch.nn as nn

from qlora_qr_svd.sam.lora import LoraSAMPerturbation
from qlora_qr_svd.sam.random_subspace import compute_random_paired_basis


class _TinyAdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(3, 4))
        self.lora_B = nn.Parameter(torch.randn(2, 3))
        self.classifier = nn.Parameter(torch.randn(2, 4))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        adapter = self.lora_B @ self.lora_A
        return inputs @ (self.classifier + adapter).T


def test_lora_sam_perturbs_only_lora_and_restores_exactly():
    torch.manual_seed(23)
    model = _TinyAdapterModel()
    manager = LoraSAMPerturbation(model, rho=0.2)
    inputs = torch.randn(5, 4)
    target = torch.randn(5, 2)
    original_lora = [model.lora_A.detach().clone(), model.lora_B.detach().clone()]
    original_classifier = model.classifier.detach().clone()

    manager.begin_clean_pass()
    clean_loss = (model(inputs) - target).square().mean()
    gradients = torch.autograd.grad(clean_loss, manager.clean_tensors())
    diagnostics = manager.begin_perturbed_pass(gradients)

    assert abs(diagnostics["perturbation_norm"] - 0.2) < 1e-6
    assert not torch.equal(model.lora_A, original_lora[0])
    assert not torch.equal(model.lora_B, original_lora[1])
    assert torch.equal(model.classifier, original_classifier)

    manager.disable()
    assert torch.equal(model.lora_A, original_lora[0])
    assert torch.equal(model.lora_B, original_lora[1])
    assert torch.equal(model.classifier, original_classifier)


def test_random_paired_basis_is_orthonormal_and_reproducible():
    torch.manual_seed(999)
    global_rng_before = torch.random.get_rng_state()
    first_generator = torch.Generator().manual_seed(12345)
    second_generator = torch.Generator().manual_seed(12345)
    first = compute_random_paired_basis(7, 5, rank=3, generator=first_generator)
    second = compute_random_paired_basis(7, 5, rank=3, generator=second_generator)

    identity = torch.eye(3)
    assert first.source == "random"
    assert torch.allclose(first.left.T @ first.left, identity, atol=1e-6)
    assert torch.allclose(first.right.T @ first.right, identity, atol=1e-6)
    assert torch.equal(first.left, second.left)
    assert torch.equal(first.right, second.right)
    assert torch.equal(torch.random.get_rng_state(), global_rng_before)

    dense_modes = [
        first.left[:, index : index + 1] @ first.right[:, index : index + 1].T
        for index in range(first.rank)
    ]
    gram = torch.stack(
        [
            torch.stack([(left * right).sum() for right in dense_modes])
            for left in dense_modes
        ]
    )
    assert torch.allclose(gram, identity, atol=1e-6)
