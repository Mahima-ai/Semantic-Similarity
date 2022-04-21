FROM python:3.8
RUN apt-get -y install libc-dev
# RUN apt-get -y install build-essential
RUN pip install -U pip

RUN mkdir /BLOG
ADD . /BLOG
WORKDIR /BLOG
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "main_page.py"]