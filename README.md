# landmarks-search-app

# How to run
## Install requirements (for example, via poetry)
poetry install

## Locally
streamlit run landmarks_app.py

## With Docker (check that CUDA available or use another base image)
docker compose up (--build)

# Download checkpoints
- [DeiT Roadmap on 10k landmarks](https://drive.google.com/file/d/1VNXP6X7YCUmzR9QGnm0Gwxy6MW3DHrOw/view?usp=share_link)

# Metrics
## Image recognition 10k
|Model-backbone|GAP |Accuracy|Image Resolution|FPS   |N_epochs|Time(epoch)|
|--------------|----|--------|----------------|------|--------|-----------|
|DOLG-Resnet101|0.63|0.67    |512             |N/A   |30      |~3h        |
|DOLG-Resnet101|0.54|0.58    |256             |N/A   |30      |~1h        |
|DOLG-EffnetB0 |0.65|0.68    |512             |N/A   |30      |~1h        |

## Image retrieval 10k 
|Model-backbone   |Accuracy |GAP |R@1 |R@2 |R@10|Image Resolution|FPS   |N_epochs|Time(epoch)|
|-----------------|---------|----|----|----|----|----------------|------|--------|-----------|
|ResNet18-ROADMAP |0.67     |0.58|0.64|0.69|0.78|224             |371.2 |1000    |~5 min   |
|DeiT-ROADMAP     |0.80     |0.73|0.76|0.79|0.85|224             |156.7 |1000    |~5 min     |
|DOLG-EffnetB0    |0.69     |0.58|0.69|0.78|0.91|512             |N/A   |30      |~1 h       |

### ArcFace (нужно посчитать метрики на том же наборе данных и тем же способом, что и в таблице выше, для верного сравнения)
|Model-backbone  |MIN_IMG  |IMG_S|B_S|P@1 |P@3 |P@5 |R@1 |R@3 |R@5 |N_ep|Time(ep)|
|----------------|---------|-----|---|----|----|----|----|----|----|----|--------|
|ResNet50 (3 frz)|30 per cl|256  |64 |0.55|0.47|0.43|0.26|0.66|1.0 |4   |11 min  |
|ResNet50 (3 frz)|30 per cl|256  |64 |0.64|0.57|0.54|0.24|0.64|1.0 |15  |11 min  |
|ResNet50 (3 frz)|30 per cl|512  |24 |0.41|0.34|0.30|0.15|0.40|0.6 |4   |43 min  |
|ResNet50 (3 frz)|30 per cl|512  |24 |0.64|0.55|0.50|0.14|0.38|0.58|15  |43 min  |
|ResNet50 (3 frz)|30 per cl|256  |64 |0.86|0.83|0.82|0.11|0.31|0.52|15  |4.5 min |


### DELF Experiments
| Classes 	| Model 	| Img_size 	| GAP   	| MAP   	| MRR@1 	| MRR@3 	| MRR@5 	| P@1 	| P@3 	| P@5 	| RAM required 	|
|-----------	|-------	|----------	|-------	|-------	|-------	|-------	|-------	|-------------	|-------------	|-------------	|--------------	|
| 7003      	| EffB0 	| 256      	| 0.633 	| 0.857 	| 0.854 	| 0.869 	| 0.876 	| 0.854       	| 0.738       	| 0.644       	| 32 Gb        	|
| 2973      	| EffB0 	| 224      	| 0.705 	| 0.896 	| 0.900 	| 0.909 	| 0.913 	| 0.900       	| 0.799       	| 0.716       	| 24 Gb        	|
| 1991      	| EffB0 	| 224      	| 0.715 	| 0.892 	| 0.896 	| 0.903 	| 0.907 	| 0.896       	| 0.801       	| 0.724       	| 16 Gb        	|
| 497       	| EffB0 	| 224      	| 0.812 	| 0.918 	| 0.910 	| 0.923 	| 0.929 	| 0.910       	| 0.870       	| 0.821       	| < 8 Gb         	|
| 7003      	| EffB7 	| 256      	| 0.473 	| 0.597 	| 0.601 	| 0.607 	| 0.610 	| 0.601       	| 0.556       	| 0.531       	| 32 Gb        	|
| 2973      	| EffB7 	| 256      	| 0.548 	| 0.643 	| 0.648 	| 0.651 	| 0.652 	| 0.648       	| 0.613       	| 0.590       	| 24 Gb        	|
| 1991      	| EffB7 	| 256      	| -     	| 0.824 	| 0.840 	| 0.852 	| 0.857 	| 0.840       	| 0.727       	| 0.639       	| 16 Gb        	|
| 497       	| EffB7 	| 256      	| -     	| 0.851 	| 0.847 	| 0.855 	| 0.861 	| 0.847       	| 0.823       	| 0.811       	| < 8 Gb         	|
| 249       	| EffB7 	| 512      	| -     	| 0.969 	| 0.972 	| 0.976 	| 0.977 	| 0.972       	| 0.935       	| 0.880       	| < 8 Gb       	|
| 102       	| EffB7 	| 512      	| -     	| 0.971 	| 0.970 	| 0.975 	| 0.976 	| 0.970       	| 0.951       	| 0.926       	| < 8 Gb       	|
