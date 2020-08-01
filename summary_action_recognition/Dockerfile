FROM jjanzic/docker-python3-opencv:latest
WORKDIR /usr/src/app
COPY Summary .
COPY Action-Recognition .
COPY script.sh .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["bash","script.sh"]