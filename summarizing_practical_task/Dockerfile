FROM python:3.12-alpine

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY summarizing_practical_task/ .

EXPOSE 5000

CMD ["python", "api.py"]
# ENTRYPOINT ["python", "api.py"]
