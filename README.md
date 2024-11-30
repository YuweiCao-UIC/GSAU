This repository contains code and configuration files fro reproducing the experimental results in our paper, "Graph-Sequential Alignment and Uniformity: Toward Enhanced Recommendation Systems" (under review).

All methods are implemented using the [RecBole](https://recbole.io/) framework, except those already supported.

### Run GSAU (rec)
```
python run_gsau.py --enable_graph_encoder --enable_sequential_encoder \
  --sequential_encoder=SASRec --enable_u_i_uniformity --rec --gamma=0.1 \
  --dataset=amazon-toys-games
```

### Run GSAU
```
python run_gsau.py --enable_graph_encoder --enable_sequential_encoder \
  --sequential_encoder=SASRec --enable_u_i_uniformity --gamma=0.1 \
  --dataset=amazon-toys-games
```

### Run SASRec (u)
```
python run_gsau.py --enable_sequential_encoder --sequential_encoder=SASRec \
  --enable_u_i_uniformity --gamma=0.05 --dataset=amazon-toys-games
```

### Run BERT4Rec (u)
```
python run_gsau.py --enable_sequential_encoder --sequential_encoder=BERT4Rec \
  --enable_u_i_uniformity --gamma=0.05 --dataset=amazon-toys-games
```

### Run DirectAU
```
python run_gsau.py --enable_graph_encoder --predict_with_graph_encoder \
  --gamma=0.5 --dataset=amazon-toys-games
```

### Run GraphAU
```
python run_gsau.py --enable_graph_encoder --predict_with_graph_encoder \
  --align_per_layer --gamma=0.5 --dataset=amazon-toys-games
```

### Run LightGCN/SASRec/BERT4Rec
```
python run_baselines.py
```
