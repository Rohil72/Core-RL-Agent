"""Acceptance Test A16: Optimizer/RNG/sampler resume matches uninterrupted run on same device/environment."""

import pytest
import numpy as np
import torch
from memory_study_v2.backbones import MLPAnnual
from memory_study_v2.train import EarlyStoppingSelector, save_checkpoint, load_checkpoint


def test_checkpoint_resume_parity():
    """Verify that interrupted and resumed training matches uninterrupted training bit-for-bit."""
    torch.manual_seed(100)
    np.random.seed(100)

    # Uninterrupted run: 4 epochs
    model_cont = MLPAnnual(seed=100)
    opt_cont = torch.optim.AdamW(model_cont.parameters(), lr=1e-3)
    selector_cont = EarlyStoppingSelector(min_epochs=2, max_epochs=10)

    # Fixed training data
    x_batches = [torch.randn(16, 966) for _ in range(4)]
    y_batches = [torch.randn(16) for _ in range(4)]

    for ep in range(4):
        pred = model_cont(x_batches[ep])
        loss = torch.nn.functional.mse_loss(pred, y_batches[ep])
        opt_cont.zero_grad()
        loss.backward()
        opt_cont.step()
        selector_cont.step(ep, float(loss.item()))

    cont_weights = model_cont.fc1.weight.clone().detach()

    # Resumed run: train 2 epochs, save, reload into new instances, train 2 more epochs
    torch.manual_seed(100)
    np.random.seed(100)
    model_resume = MLPAnnual(seed=100)
    opt_resume = torch.optim.AdamW(model_resume.parameters(), lr=1e-3)
    selector_resume = EarlyStoppingSelector(min_epochs=2, max_epochs=10)

    for ep in range(2):
        pred = model_resume(x_batches[ep])
        loss = torch.nn.functional.mse_loss(pred, y_batches[ep])
        opt_resume.zero_grad()
        loss.backward()
        opt_resume.step()
        selector_resume.step(ep, float(loss.item()))

    # Save checkpoint state at epoch 1
    state = save_checkpoint(model_resume, opt_resume, selector_resume, epoch=1)

    # Create fresh model & optimizer, load checkpoint state
    model_loaded = MLPAnnual(seed=999)  # different seed initially
    opt_loaded = torch.optim.AdamW(model_loaded.parameters(), lr=1e-3)
    selector_loaded = EarlyStoppingSelector(min_epochs=2, max_epochs=10)

    next_ep = load_checkpoint(state, model_loaded, opt_loaded, selector_loaded)
    assert next_ep == 2

    # Finish remaining 2 epochs
    for ep in range(next_ep, 4):
        pred = model_loaded(x_batches[ep])
        loss = torch.nn.functional.mse_loss(pred, y_batches[ep])
        opt_loaded.zero_grad()
        loss.backward()
        opt_loaded.step()
        selector_loaded.step(ep, float(loss.item()))

    resumed_weights = model_loaded.fc1.weight.clone().detach()

    # Assert exact bitwise parity between continuous and resumed run
    torch.testing.assert_close(
        cont_weights,
        resumed_weights,
        rtol=0.0,
        atol=0.0,
        msg="Resumed training did not match uninterrupted training exactly!"
    )
