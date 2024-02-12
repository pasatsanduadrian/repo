
# Rularea unei Aplicații Streamlit în Google Colab și Accesarea Acesteia Prin `ngrok`

Acest ghid explică cum să rulați o aplicație Streamlit în Google Colab și să o faceți accesibilă public prin `ngrok`.

## Despre Proiect

Acest proiect demonstrează cum să transformați un script Python într-o aplicație web interactivă folosind Streamlit, să o executați în mediul cloud Google Colab și să o expuneți pe internet prin `ngrok`.

## Pași de Urmărit

### 1. Pregătirea Mediului Google Colab

- Deschideți un nou notebook în [Google Colab](https://colab.research.google.com/).
- Asigurați-vă că rulați notebook-ul cu suport Python.

### 2. Instalarea Dependințelor

Instalați Streamlit și `pyngrok` folosind `pip`:

```bash
!pip install streamlit pyngrok
```

### 3. Crearea Scriptului Streamlit

Crearea fișierului pentru scriptul Streamlit (exemplu: `app.py` sau `script.py`) în notebook:

```python
%%writefile app.py
import streamlit as st

st.title('Titlul Aplicației')
# Adăugați restul codului aplicației aici
```

### 4. Rularea Aplicației Streamlit

Lansați aplicația Streamlit în background folosind comanda:

```bash
!streamlit run app.py &>/dev/null&
```

### 5. Configurarea și Conectarea `ngrok`

Configurați `ngrok` cu authtoken-ul dvs. și creați un tunel către portul aplicației Streamlit:

```python
from pyngrok import ngrok

ngrok.set_auth_token('AUTHTOKEN_AICI')
public_url = ngrok.connect(addr="8501")
print(public_url)
```

### 6. Accesarea Aplicației

Folosiți URL-ul afișat de `ngrok` pentru a accesa aplicația Streamlit din orice browser.

## Notă

Aplicația Streamlit va fi accesibilă atât timp cât sesiunea Colab este activă. Pentru a opri tunelul `ngrok`, folosiți:

```python
ngrok.disconnect(public_url)
```
