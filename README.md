# 3D voxelデータから拡散率と透過率を予測する
### 目標
すべてのコードを自力でシンプルに書き上げて、MLOpsのすべてを自力実装する。


### レギュレーション
 - AIなどに質問するのは可能、ただしコード生成はさせないこと。
 - 自力で書いた後にどう修正したほうが良いかアドバイスをもらうのは可能。
 - Pythonなどの知識を得るために、寄り道をしまくること
 - どうしたら効率が良くなるか、どのライブラリを入れたらよりよいかなどいちいち考えること、調べること
 - ゆっくりやること、焦らないこと、コピーなどは極力せずに書くこと


### Pythonの仮想環境
uvの仮想環境作成
uv venv .venv

uvのライブラリインポート
uv pip install -r requirements.txt

.venvのライブラリのリスト
uv pip list

*.pyの実行
uv run *.py


### Gitlab runnerの登録方法
gitlab runnerをdocker-composeで動かす場合
まずdocker-composeで作成し、そこのgitlabのなかでrunnerの登録を行う。
docker exec -it gitlab-runner gitlab-runner register

Enter the GitLab instance URL: https://gitlab.com/
Enter the registration token: <GitLabで取得したトークン>
Enter a description for the runner: docker-runner
Enter tags for the runner (comma-separated): 
Enter executor: docker
Enter default Docker image (eg. ruby:2.7): ojt-project:v1.0

config.tomlはvolumeでpwd上で永続化されているので、docker-compose downしても保存されている。

アクセストークンの取得方法は
設定→CI/CD→Runner→プロジェクトのrunnerが無料版だが、企業版だと不明


### チームで開発をしている意識をもって
Gitは一つの機能を追加するとき、ブランチを切る
コードの保守がしやすいようなコード設計にする
コードの要件定義のやり方を覚えたほうが良い


### 気になるキーワード
 - Weights and biases
 - DVC
 - LakeFS
 - TensorBoard
 - wandb
 - tqdm
 - mとkappa片方だけ予測
 - uv


### Dockerfile or docker-compose を編集した場合
docker build -t ojt:v1.0 ./local_image_Dockerfile/
docker compose build
