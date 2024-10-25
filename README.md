<div align="center">
  <h1 align="center">COALA: A Practical and Vision-centric Federated Learning Platform (ICML24: https://openreview.net/pdf?id=ATRnM8PyQX)</h1>

[📘 Documentation](https://coalafl.netlify.app/) | [🛠️ Installation](https://coalafl.netlify.app/get_started.html)
</div>

## Introduction

COALA is a practical and vision federated learning (FL) platform, and a suite of benchmarks for practical FL scenarios, which we categorize into three levels: task, data, and model.

<img src="images/coala-overview.png" width="700">

At the task level, COALA extends support from simple classification to 15 computer vision tasks, including object detection, segmentation, pose estimation, and more. It also facilitates federated multiple-task learning, allowing clients to tackle multiple tasks simultaneously. 

At the data level, COALA goes beyond supervised FL to benchmark both semi-supervised FL and unsupervised FL. It also benchmarks feature distribution shifts other than commonly considered label distribution shifts. In addition to dealing with static data, it supports federated continual learning for continuously changing data in real-world scenarios. 

At the model level, COALA benchmarks FL with split models and different models in different clients. 

COALA platform offers three degrees of customization for practical FL scenarios, including configuration customization, components customization, and workflow customization.

It enables users with various levels of expertise to experiment and prototype FL applications with little/no coding. It aims to support production-level deployment of FL applications for a wide-range of business use cases, such as smart city, smart retail, and smart factory applications.

You can use it for:
* FL Research on algorithm and system
* Proof-of-concept (POC) of new FL applications
* Production-level FL applications


## Getting Started

You can refer to [Get Started](docs/en/get_started.md) for installation and [Quick Run](docs/en/quick_run.md) for the simplest way of using COALA.

For more advanced usage, we provide a list of tutorials on:
* [High-level APIs](docs/en/tutorials/high-level_apis.md)
* [Configurations](docs/en/tutorials/config.md)
* [Datasets](docs/en/tutorials/dataset.md)
* [Models](docs/en/tutorials/model.md)
* [Customize Server and Client](docs/en/tutorials/customize_server_and_client.md)
* [Distributed Training](docs/en/tutorials/distributed_training.md)


# License

This project is released under the [Apache 2.0 license](LICENSE).

# Citation

If you use this platform or related projects in your research, please cite this paper.

```
@inproceedings{zhuangcoala,
  title={COALA: A Practical and Vision-Centric Federated Learning Platform},
  author={Zhuang, Weiming and Xu, Jian and Chen, Chen and Li, Jingtao and Lyu, Lingjuan},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
