FROM pytorch/torchserve:latest
USER root
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0 

USER model-server
COPY ./resources/ /home/model-server/resources/
RUN pip install -r /home/model-server/resources/requirements.txt

EXPOSE 8080 8081

RUN torch-model-archiver \
    --model-name yolov5 \
    --version 0.1 \
    --serialized-file /home/model-server/resources/yolov5n.torchscript \
    --export-path /home/model-server/model-store \
    --handler /home/model-server/resources/yolov5_handler.py \
    --extra-files /home/model-server/resources/index_to_name.json,/home/model-server/resources/yolov5_handler.py

CMD [ "torchserve", \
    "--start", \
    "--model-store", \
    "/home/model-server/model-store", \
    "--models", \
    "yolov5=yolov5.mar" ]