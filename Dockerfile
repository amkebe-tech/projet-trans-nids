# Image de base légère avec Python 3.10 (version minimale requise par le
# cahier des charges, section 3.2).
FROM python:3.10-slim

# Dossier de travail à l'intérieur du conteneur.
WORKDIR /app

# On copie d'abord uniquement requirements.txt pour profiter du cache Docker :
# si le code change mais pas les dépendances, cette étape n'est pas rejouée.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On copie ensuite le code de l'application.
# (models/, notebooks/, les .csv/.parquet/.pth etc. sont exclus via
# .dockerignore et .gitignore : ce sont de gros fichiers qui n'ont pas leur
# place dans l'image Docker du prototype.)
COPY app/ ./app/

# Port par défaut de Streamlit.
EXPOSE 8501

# Lancement de l'application. --server.address=0.0.0.0 est nécessaire pour
# que le serveur soit accessible depuis l'extérieur du conteneur.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
