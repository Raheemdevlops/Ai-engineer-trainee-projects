FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY model/ model/

EXPOSE 5000

CMD ["python", "src/app.py"]
