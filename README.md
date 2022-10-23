# landmarks-search-app


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
