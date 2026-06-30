# Badanie LLM

Zestaw lokalnych aplikacji webowych (Streamlit) do pracy z modelami AI przez ich oficjalne API:

- **`app.py`** – multiczat tekstowy (OpenAI, Grok/xAI, DeepSeek) z pełną kontrolą nad System Promptem.
- **`image_app.py`** – generator obrazów (OpenAI, Gemini, Grok/xAI).

---

## Wymagania

- **Python 3.9+** ([python.org/downloads](https://www.python.org/downloads/) – przy instalacji zaznacz **„Add Python to PATH”**)
- Klucze API dostawców, z których chcesz korzystać.

---

## Instalacja (krok po kroku)

Otwórz terminal (PowerShell) w folderze projektu i wykonaj kolejno:

### 1. Przejdź do folderu projektu
```powershell
cd C:\Users\TWOJA_NAZWA\Documents\Badanie_LLM
```

### 2. Utwórz i aktywuj środowisko wirtualne
```powershell
python -m venv venv
venv\Scripts\activate
```
Po aktywacji zobaczysz `(venv)` na początku linii.

> **macOS / Linux:** zamiast `venv\Scripts\activate` użyj `source venv/bin/activate`.

### 3. Zainstaluj biblioteki
```powershell
pip install -r requirements.txt
```

### 4. Skonfiguruj klucze API
```powershell
copy .env.example .env
notepad .env
```
W pliku `.env` wpisz swoje klucze (wystarczy ten dostawca, którego chcesz używać), zapisz i zamknij:

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
XAI_API_KEY=xai-...
DEEPSEEK_API_KEY=sk-...
```

> **macOS / Linux:** zamiast `copy` użyj `cp .env.example .env`.

---

## Uruchamianie

> Przy każdym nowym oknie terminala najpierw aktywuj środowisko: `venv\Scripts\activate`

### Aplikacja czatu (tekst)
```powershell
streamlit run app.py
```

### Generator obrazów
```powershell
streamlit run image_app.py
```

Po uruchomieniu przeglądarka otworzy się automatycznie pod adresem **http://localhost:8501**.
Aby zatrzymać aplikację, wciśnij `Ctrl + C` w terminalu.

> Jeśli polecenie `streamlit` nie jest rozpoznawane, użyj: `python -m streamlit run app.py`.

---

## Skąd wziąć klucze API

| Dostawca | Strona |
|----------|--------|
| OpenAI   | https://platform.openai.com/api-keys |
| Gemini   | https://aistudio.google.com/app/apikey |
| Grok (xAI) | https://console.x.ai |
| DeepSeek | https://platform.deepseek.com/api_keys |

---

## Najczęstsze problemy

- **`streamlit: not recognized`** – nie aktywowałeś środowiska (`venv\Scripts\activate`) lub biblioteki nie są zainstalowane. Spróbuj `python -m streamlit run app.py`.
- **`python: not recognized`** – Python nie jest zainstalowany lub nie dodano go do PATH (zainstaluj ponownie z zaznaczoną opcją „Add Python to PATH”).
- **Błąd 401 / „Invalid API key”** – zły lub pusty klucz w pliku `.env`.
- **Błąd 429 / „insufficient_quota”** (OpenAI) lub **402 / „Insufficient Balance”** (DeepSeek) – brak środków na koncie, doładuj je u dostawcy.
- **„Model not available” / „not found”** – wybrany model nie jest dostępny dla Twojego konta albo zmieniło się jego ID. W generatorze obrazów możesz wpisać własne ID w polu **Custom model**.
