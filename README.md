# landmarks-search-app


Project Organization
```
├── api                             <- Core api service
│   ├── checkpoints                 <- Checkpoints and artifacts
│   ├── retrieval                   <- Model engine
│   ├── src                         <- Inference stuff
│   ├── api.py
│   ├── Dockerfile                  <- If you want to run just api part
│   └── pyproject.toml              <- Need for API requirements
|
├── bot                             <- Telegram bot service
│   ├── bot.env                     <- Credentials
│   ├── bot.py
│   ├── Dockerfile                  <- If you want to run just bot part
│   └── docker-compose.yml          <- If you want to run just bot part
|
├── web                             <- Web interface service
│   ├── .streamlit                  <- Customization
│   ├── web.env                     <- Credentials
│   ├── web.py
│   ├── Dockerfile                  <- If you want to run just web part
│   └── docker-compose.yml          <- If you want to run just web part
|
├── data                            <- Common data for all services
│   ├── filtered                    <- Filtered clean dataset
│   └── train10k                    <- 10k classes subset
|
├── .dockerignore
├── .gitignore
├── .python-version
├── docker-compose.yml
├── poetry.lock
├── pyproject.toml
└── README.md
```


# How to run
## Download dataset
- [Filtered clean dataset](https://cloud.mail.ru/public/ZsDS/jJkPRKKDd)

## Download checkpoints
- [DeiT Roadmap on 10k landmarks](https://cloud.mail.ru/public/FiL9/fa79wiv78)

## Run with Docker (check that CUDA available or use another base image)
```
docker compose up (--build)
```


# Metrics
|Model-backbone      |SIZE |ACC |GAP |R@1 |R@2 |R@10|Epoch time|FPS  |
|--------------------|-----|----|----|----|----|----|----------|-----|
|DELF + EffnetB0*    |256  |0.85|0.63|0.85|0.87|0.88|      ~1 h|  N/A|
|DOLG + EffnetB0     |512  |0.69|0.58|0.69|0.78|0.91|      ~1 h|  N/A|
|ResNet + ArcFace    |256  |0.70|0.62|0.65|0.72|0.74|      ~9 m|342.2|
|ResNet + ROADMAP    |224  |0.67|0.58|0.64|0.69|0.78|      ~5 m|371.2|
|DeiT + ROADMAP      |224  |0.80|0.73|0.76|0.79|0.85|      ~5 m|156.7|
|CLIP openai/ViT-B-16|224  |0.65|0.5 |0.6 |0.68|0.76|       N/A|129.5|
|Best                |     |0.80|0.73|0.76|0.79|0.91|          |     |

