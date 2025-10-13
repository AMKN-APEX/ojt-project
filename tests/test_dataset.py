import os
import torch

from src.dataset import PorousDataset


def _make_pt(path, tensor):
    """小さなテンソルを .pt ファイルに保存するユーティリティ。"""
    torch.save(tensor, path)


def test_porousdataset_basic(tmp_path):
    """基本動作: 一時ディレクトリ上でファイルを作成し、読み込みと順序をチェックする。"""
    # テスト用ディレクトリを作る
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # X ファイルを作成 (structure_1.pt, ... , structure_10.pt)
    for i in range(10):
        x_path = data_dir / f"structure_{i+1}.pt"
        _make_pt(str(x_path), torch.randn(3, 32, 32))

    # y ファイルを作成（X_dir の外に置く）
    y = torch.arange(10, dtype=torch.float32)
    y_path = tmp_path / "m_train.pt"
    _make_pt(str(y_path), y)

    # データセットを初期化
    ds = PorousDataset(str(data_dir), str(y_path), nums_data=10)

    # 長さとファイル順をチェック
    assert len(ds) == 10
    basenames = [os.path.basename(p) for p in ds.X_files]
    assert basenames == [f"structure_{i+1}.pt" for i in range(10)]

    # __getitem__ の戻り値の型を確認
    x0, y0 = ds[0]
    assert isinstance(x0, torch.Tensor)
    assert isinstance(y0, torch.Tensor)


def test_porousdataset_truncate(tmp_path):
    """nums_data によるトランケート処理を確認する。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # 5 個の X ファイルを作成
    for i in range(5):
        x_path = data_dir / f"structure_{i+1}.pt"
        _make_pt(str(x_path), torch.randn(3, 16, 16))

    # y ファイル（X_dir の外に置く）
    y = torch.arange(5, dtype=torch.float32)
    y_path = tmp_path / "m_train.pt"
    _make_pt(str(y_path), y)

    # nums_data=3 により 3 サンプルに制限されるはず
    ds = PorousDataset(str(data_dir), str(y_path), nums_data=3)
    assert len(ds) == 3
    basenames = [os.path.basename(p) for p in ds.X_files]
    assert basenames == ["structure_1.pt", "structure_2.pt", "structure_3.pt"]
    

