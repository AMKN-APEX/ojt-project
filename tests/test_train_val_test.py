import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pytest
from unittest.mock import patch

from src.train_val_test import TrainValTest

# テスト用のシンプルなモデル
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# テストデータとコンポーネントを準備するためのフィクスチャ
@pytest.fixture
def setup_train_val_test():
    # ダミーデータとDataLoaderの作成
    X = torch.randn(20, 10)
    y1 = torch.randn(20, 1)
    y2 = torch.randn(20, 1)

    # train_val_test.pyの実装に合わせて、yをタプルとして扱うカスタムデータセット
    class CustomDataset(TensorDataset):
        def __init__(self, *tensors):
            super().__init__(*tensors)
        
        def __getitem__(self, index):
            # x, y1, y2 を取得
            tensors = super().__getitem__(index)
            # (x, (y1, y2)) の形式で返す
            return tensors[0], (tensors[1], tensors[2])

    dataset = CustomDataset(X, y1, y2)
    loader = DataLoader(dataset, batch_size=4)

    # モデル、損失関数、オプティマイザのインスタンス化
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    # TrainValTestクラスのインスタンスを返す
    return TrainValTest(
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=2
    )

# run_epochメソッドのテスト
def test_run_epoch(setup_train_val_test):
    tvt = setup_train_val_test
    
    # 訓練モードでの実行
    initial_params = [p.clone() for p in tvt.model.parameters()]
    train_loss1, train_loss2 = tvt.run_epoch(tvt.train_loader, train=True)
    
    # 損失が計算されていることを確認
    assert isinstance(train_loss1, float)
    assert isinstance(train_loss2, float)
    assert train_loss1 >= 0.0
    assert train_loss2 >= 0.0

    # 訓練モードではパラメータが更新されることを確認
    params_updated = False
    for p_initial, p_final in zip(initial_params, tvt.model.parameters()):
        if not torch.equal(p_initial, p_final):
            params_updated = True
            break
    assert params_updated

    # 評価モードでの実行
    initial_params_eval = [p.clone() for p in tvt.model.parameters()]
    val_loss1, val_loss2 = tvt.run_epoch(tvt.val_loader, train=False)

    # 損失が計算されていることを確認
    assert isinstance(val_loss1, float)
    assert isinstance(val_loss2, float)

    # 評価モードではパラメータが更新されないことを確認
    params_not_updated = True
    for p_initial, p_final in zip(initial_params_eval, tvt.model.parameters()):
        if not torch.equal(p_initial, p_final):
            params_not_updated = False
            break
    assert params_not_updated

# train_valメソッドのテスト
@patch('src.train_val_test.mlflow')
def test_train_val(mock_mlflow, setup_train_val_test):
    tvt = setup_train_val_test
    tvt.train_val()

    # mlflow.log_metricが期待通りに呼び出されたか確認
    assert mock_mlflow.log_metric.call_count == tvt.num_epochs * 4
    mock_mlflow.log_metric.assert_any_call("train_loss_m", pytest.approx(0, abs=1e4), step=0)
    mock_mlflow.log_metric.assert_any_call("train_loss_k", pytest.approx(0, abs=1e4), step=0)
    mock_mlflow.log_metric.assert_any_call("val_loss_m", pytest.approx(0, abs=1e4), step=0)
    mock_mlflow.log_metric.assert_any_call("val_loss_k", pytest.approx(0, abs=1e4), step=0)

# testメソッドのテスト
@patch('src.train_val_test.mlflow')
def test_test(mock_mlflow, setup_train_val_test):
    tvt = setup_train_val_test
    tvt.test()

    # mlflow.log_metricが期待通りに呼び出されたか確認
    assert mock_mlflow.log_metric.call_count == 2
    mock_mlflow.log_metric.assert_any_call("test_loss_m", pytest.approx(0, abs=1e4))
    mock_mlflow.log_metric.assert_any_call("test_loss_k", pytest.approx(0, abs=1e4))
