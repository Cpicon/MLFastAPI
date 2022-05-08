#
FROM python:3.9

RUN pip install --upgrade pip
#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./app /code/app
# download the model so the whn the image is ready, restarting the container will be quick
RUN ["python","app/download_model.py"]

EXPOSE 8000
#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
