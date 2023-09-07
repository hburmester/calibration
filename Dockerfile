FROM python:3.8

ADD main.py .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install opencv-python matplotlib numpy cv

CMD ["python", "./main.py"]
