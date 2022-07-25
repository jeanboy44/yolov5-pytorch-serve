# To Serve Model
## 0. Install yolov5
```sh
pip install yolov5
```
## 1. Download model
```sh
python src/serve/models/yolov5/download_model.py "src/serve/models/yolov5/resources/yolov5n.pt"
```
## 2. Export model to torchscript
```sh
yolov5 export --weights "src/serve/models/yolov5/resources/yolov5n.pt" --include 'torchscript,' --batch_size 1
```

## 3. Build Image
```
docker build src/serve/models/yolov5 -t torchserve/yolov5
```

## 4. Run Torchserve
```
docker run -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 -v $PWD/src/serve/models/yolov5/resources:/home/model-server/resources --name "torchserve" torchserve/yolov5:latest
```

# Test API
## 1. Check health
```{sh}
curl http://localhost:8080/ping
```

## 2. Predict
```{sh}
curl http://localhost:8080/predictions/yolov5 -T src/serve/models/examples/images/000000000009.jpg
```