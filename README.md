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
|ResNet50 (3 frz)|30 per cl|256  |64 |0.86|0.83|0.82|0.11|0.31|0.52|15  |4.5 min |


### DELF Experiments
**DELF MODEL Accuracy**
| Number of Classess | @1 | @3 | @5 |
| ------------------ | -- | -- | -- |
|7003|0.601| 0.556| 0.482|
|2971|0.648| 0.613| 0.555|
| 1991| 0.84|0.727|0.483|
|497|0.847| 0.823| 0.811|
|249|	0.972|	0.935|	0.880|
|102 |	0.970	|0.951|	0.926|

**DELF MODEL Precision_k**
| Number of Classess | @1 | @3 | @5 |
| ------------------ | -- | -- | -- |
|7003|0.601| 0.556| 0.531|
|2971|0.648| 0.613| 0.59|
| 1991| 0.84|	0.727	|0.639|
|497|0.847| 0.823| 0.811|
|249|	0.972 |	0.935|	0.880|
|102 |	0.970 |	0.951|	0.926|

**DELF MODEL Recall_k:**
| Number of Classess | @1 | @3 | @5 |
| ------------------ | -- | -- | -- |
|7003|0.124| 0.268| 0.381|
|2971| 0.114| 0.262| 0.382|
| 1991| 0.249|	0.515|	0.668|
|497|0.198| 0.539| 0.871|
|249|	0.237|	0.649|	0.978|
|102 |	0.218 |	0.615|	0.978|

**DELF MODEL MRR Score:**
| Number of Classess | @1 | @3 | @5 |
| ------------------ | -- | -- | -- |
|7003|0.601| 0.607| 0.61|
|2971|0.648| 0.651| 0.652|
| 1991| 0.840|	0.852|	0.857|
|497|0.847| 0.855| 0.861|
|249|	0.972|	0.976|	0.977|	0.969|
|102 |	0.970|	0.975|	0.976|


**DELF MODEL MAP Score:**
| Number of Classess | |
| ------------------ | -- | 
|7003|0.597|
|2971|0.643|
| 1991| 0.824|
|497| 0.851|
|249|	0.969|	
|102 |	0.971|	

**DELF MODEL GAP Score:**
| Number of Classess | |
| ------------------ | -- | 
| 7003| 0.473|
|2971|	0.548|	
	