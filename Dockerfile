# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Imposta la directory di lavoro nel container
WORKDIR /app

# Copia tutto il codice nella directory di lavoro nel container
COPY . /app

# Installa le dipendenze elencate nel file requirements.txt
RUN pip install -r requirements.txt

# Scarica i file richiesti
RUN wget -q --show-progress --progress=bar:force:noscroll -O ckpt/densepose/model_final_162be9.pkl https://huggingface.co/camenduru/IDM-VTON/resolve/main/densepose/model_final_162be9.pkl && \
    wget -q --show-progress --progress=bar:force:noscroll -O ckpt/humanparsing/parsing_atr.onnx https://huggingface.co/camenduru/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx && \
    wget -q --show-progress --progress=bar:force:noscroll -O ckpt/humanparsing/parsing_lip.onnx https://huggingface.co/camenduru/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx && \
    wget -q --show-progress --progress=bar:force:noscroll -O ckpt/openpose/ckpts/body_pose_model.pth https://huggingface.co/camenduru/IDM-VTON/resolve/main/openpose/ckpts/body_pose_model.pth
    
RUN python -u gradio_demo/build.py 
    
# Specifica il comando da eseguire quando il container viene avviato
CMD ["python", "-u", "gradio_demo/app.py"]



