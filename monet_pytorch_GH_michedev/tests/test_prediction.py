import torch

from monet_pytorch import Monet


@torch.no_grad()
def test_forward_monet(dataset_width=64, dataset_height=64):
    monet = Monet.from_config(dataset_width=dataset_width, dataset_height=dataset_height).eval()

    dummy_batch = torch.randn(32, 3, dataset_width, dataset_height)

    pred = monet(dummy_batch)

    assert isinstance(pred, dict)
    assert sorted(['z', 'loss', 'neg_log_p_x',
                   'kl_mask', 'kl_latent', 'mask',
                   'slot', 'mask_pred', 'log_mask_pred',
                   'log_mask']) == sorted(pred.keys())
    print("Test passed")


@torch.no_grad()
def test_forward_monet_tetrominoes():
    monet = Monet.from_config(dataset='tetrominoes').eval()

    dummy_batch = torch.randn(32, 3, 32, 32)

    pred = monet(dummy_batch)

    assert isinstance(pred, dict)
    assert sorted(['z', 'loss', 'neg_log_p_x',
                   'kl_mask', 'kl_latent', 'mask',
                   'slot', 'mask_pred', 'log_mask_pred',
                   'log_mask']) == sorted(pred.keys())


@torch.no_grad()
def test_forward_monet_clevr_6():
    monet = Monet.from_config(dataset='clevr_6').eval()

    dummy_batch = torch.randn(32, 3, 128, 128)

    pred = monet(dummy_batch)

    assert isinstance(pred, dict)
    assert sorted(['z', 'loss', 'neg_log_p_x',
                   'kl_mask', 'kl_latent', 'mask',
                   'slot', 'mask_pred', 'log_mask_pred',
                   'log_mask']) == sorted(pred.keys())
