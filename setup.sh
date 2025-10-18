#!/bin/bash

# 現在のホストIPを取得
HOST_IP=$(hostname -I | awk '{print $1}')

# .envファイルのパス（docker-compose.ymlと同じ階層にある想定）
ENV_FILE=".env"

# .env内のHOST_IP行を上書き
sed -i "s/^HOST_IP=.*/HOST_IP=${HOST_IP}/" "$ENV_FILE"

echo "HOST_IP updated to ${HOST_IP} in ${ENV_FILE}"

# Dockerコンテナを起動
docker-compose up -d