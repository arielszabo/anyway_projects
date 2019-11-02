FROM ubuntu:19.10
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y pkg-config

WORKDIR /opt
COPY . /opt/
RUN pip3 install numpy 
RUN pip3 install -r requirements.txt
# RUN ./download_and_install.sh
# CMD python3 detect_and_blur.py
