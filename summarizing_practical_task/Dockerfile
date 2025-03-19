FROM python:3.12-alpine

WORKDIR /app

# Since using alpine, installing the necessary system dependencies manually
RUN apk add --no-cache g++ gcc libffi-dev musl-dev python3-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY summarizing_practical_task/ .

EXPOSE 5000

CMD ["python", "api.py"]
# ENTRYPOINT ["python", "api.py"]
