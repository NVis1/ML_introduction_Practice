FROM python:3.12

WORKDIR /app

COPY . .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "api.py"]
# ENTRYPOINT ["python", "api.py"]
