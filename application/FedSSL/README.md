# Federated Self-supervised Learning (FedSSL)

This application integrates the code for two papers:
- Divergence-aware Federated Self-Supervised Learning, _ICLR'2022_. [[paper]](https://openreview.net/forum?id=oVE1z8NlNe)
- Collaborative Unsupervised Visual Representation Learning From Decentralized Data, _ICCV'2021_. [[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Zhuang_Collaborative_Unsupervised_Visual_Representation_Learning_From_Decentralized_Data_ICCV_2021_paper.html)


The framework implements four self-supervised learning (SSL) methods based on Siamese networks in the federated manner:
1. BYOL
2. SimSiam
3. MoCo (MoCoV1 & MoCoV2)
4. SimCLR

## Training

You can conduct training using different FedSSL methods w/ or w/o the FedEMA method. 

> You need to save the global model for further evaluation. 

### FedEMA

Run FedEMA with auto scaler $\tau=0.7$
```shell
python applications/fedssl/main.py --task_id fedema --model byol \
      --aggregate_encoder online --update_encoder dynamic_ema_online --update_predictor dynamic_dapu \
      --auto_scaler y --auto_scaler_target 0.7
```

Run FedEMA with constant weight scaler $\lambda=1$:
```shell
python applications/fedssl/main.py --task_id fedema --model byol \
      --aggregate_encoder online --update_encoder dynamic_ema_online --update_predictor dynamic_dapu \
      --weight_scaler 1
```

### Other SSL methods
Run other FedSSL methods: 
```shell
python applications/fedssl/main.py --task_id fedbyol --model byol  \
      --aggregate_encoder online --update_encoder online --update_predictor global
```
Replace `byol` in `--model byol` with other ssl methods, including `simclr`, `simsiam`, `moco`, `moco_v2` 


## File Structure
```
├── cifar_ssl <datasets preprocessing for FedSSL>
├── fl_client.py <client implementation of FedSSL>
├── fl_server.py <server implementation of FedSSL>
├── main.py <file for start running>
├── ssl_model.py <ssl models>
├── transform.py <image transformations>
└── utils.py
```

## Citation

Most Codes are from these projects.

```
@inproceedings{zhuang2022fedema,
  title={Divergence-aware Federated Self-Supervised Learning},
  author={Weiming Zhuang and Yonggang Wen and Shuai Zhang},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=oVE1z8NlNe}
}

@inproceedings{zhuang2021fedu,
  title={Collaborative Unsupervised Visual Representation Learning from Decentralized Data},
  author={Zhuang, Weiming and Gan, Xin and Wen, Yonggang and Zhang, Shuai and Yi, Shuai},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4912--4921},
  year={2021}
}
```