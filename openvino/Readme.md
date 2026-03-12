# DinoV3 OpenVINO 使用说明

本目录包含基于 OpenVINO 的 DinoV3 推理部署代码。

## 目录结构

- `requirements.txt`：依赖库列表
- `Readme.md`：使用说明

## 环境准备

1. 安装 Python 3.10+
2. 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

## 模型推理

1. 下载并放置 Torch 模型到 `model/` 目录。
2. 运行推理脚本（以 `dinov3_vision_benchmark.py` 为例）：
    ```bash
    python dinov3_vision_benchmark.py
    ```
    ### 支持的输入参数

    `dinov3_vision_benchmark.py` 支持以下主要输入参数：
    参数说明如下（详见 `utils.py` 中 `parser_args()`）：

    - `--image`：输入图片路径，默认为 `../../000000039769.jpg`
    - `--k`：聚类类别数，默认为 5
    - `--output`：输出结果保存目录，默认为 `../../outputs/`
    - `--model_path`：模型文件或目录路径，默认为 `../../models`
    - `--loop`：性能测试循环推理次数，默认为 1
    - `--tasks`：指定执行的任务类型（如 segmentation、classification、embedding、object_discovery、depth，支持多选或全部，默认为 None）

    可通过 `python dinov3_vision_benchmark.py --help` 查看所有参数及默认值。

3. 模型转换:
    `dinov3_vision_benchmark.py` 在测试时遇到不存在的OpenVINO模型，会自动转换Torch模型到OpenVINO

## 参考

- [OpenVINO 官方文档](https://docs.openvino.ai/latest/index.html)
- DinoV3 论文与官方实现

如有问题请提交 issue。