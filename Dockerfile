# Image Python 3.11 : le modèle exige scikit-learn 1.6.1 (Python 3.10 à 3.13).
FROM python:3.11-slim

WORKDIR /app

# PyTorch en version processeur (bien plus léger que la version CUDA), puis
# les autres dépendances. requirements.txt est copié seul pour profiter du
# cache Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Code, configuration du thème, modèle V7 et journaux de démonstration.
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/
COPY models/systeme_v7.joblib ./models/systeme_v7.joblib
COPY donnees_demo/ ./donnees_demo/

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
