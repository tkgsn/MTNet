FROM python:3.8-bullseye
  
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/tkgsn/MTNet.git

WORKDIR /MTNet

COPY requirements.txt /MTNet
RUN pip3 install --no-cache-dir -r requirements.txt

# 
COPY id_rsa /root/.ssh/id_rsa
COPY id_rsa.pub /root/.ssh/id_rsa.pub
COPY config /root/.ssh/config

RUN chmod 600 /root/.ssh/id_rsa

ENV DATASET chengdu
# dataset = {chengdu, geolife}
ENV MAXSIZE 10000
# 0 -> all data
ENV SEED 0
ENV DP False
# False or True
ENV EPOCH 100


CMD ["/bin/bash", "./run_chengdu.sh"]