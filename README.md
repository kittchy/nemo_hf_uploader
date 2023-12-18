# NeMo Model push tools

Author: @kittchy

## Step 1: Prepare your Model

`*.nemo` モデルを `models/` に配置します。

## Step 2: Create Config File

`config_default.yaml` を `config.yaml` にコピーします。

```bash
cp config_default.yaml config.yaml
```

その後、`config.yaml` に必要な情報を記述します。

## Step 3: Build Docker Image

- Docker Imageをpullします。

  ```bash
  docker pull kittchy/nemo-model-push-tools:latest
  ```

- もしImageがない場合やpullできない場合はdocker imageをbuildします。
  ```bash
  docker compose build
  ```

## Step 4: Push Model

docker を使ってモデルをpushします。

```bash
docker compose up
```
