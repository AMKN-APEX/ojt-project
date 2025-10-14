import torch

from src.model import CNN


def test_cnn_forward_shape():
    """モデルの順伝播が期待される出力形状 (B,) を返すことを確認する。"""
    model = CNN()
    model.eval()

    # 任意のバッチサイズと画像サイズで動くことを確認
    x = torch.randn(8, 3, 64, 64)
    y = model(x)
    assert y.shape == (8, 2)
