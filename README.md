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

# Opcjonalnie – tryb „OpenAI thinking” w generatorze obrazów
OPENAI_THINKING_MODEL=gpt-5.5
OPENAI_REASONING_EFFORT=medium
```

> **macOS / Linux:** zamiast `copy` użyj `cp .env.example .env`.

---

## Opcjonalny tryb „thinking / reasoning” (generator obrazów)

W `image_app.py` w panelu bocznym (sekcja **🧠 Reasoning**) możesz włączyć
opcjonalny tryb rozumowania. **Domyślnie jest wyłączony** — bez niego aplikacja
działa dokładnie tak jak dotychczas.

Po zaznaczeniu **„Enable thinking / reasoning”** wybierasz **silnik rozumowania**:

- **OpenAI** — model rozumujący przez Responses API (`reasoning.summary`),
  wymaga `OPENAI_API_KEY`, konfigurowalny przez `OPENAI_THINKING_MODEL`
  (domyślnie `gpt-5.5`).
- **Gemini (Google)** — model „thinking” (np. `gemini-2.5-flash`) zwracający
  podsumowanie myślenia (`include_thoughts`), wymaga `GEMINI_API_KEY`,
  konfigurowalny przez `GEMINI_THINKING_MODEL` (domyślnie `gemini-2.5-flash`).

Niezależnie od silnika:
- model najpierw dopracowuje Twój prompt,
- obraz jest generowany na podstawie dopracowanego promptu (u dowolnego dostawcy),
- pod obrazem pojawia się sekcja **„Reasoning summary”** z oficjalnym
  podsumowaniem rozumowania i przyciskiem **Kopiuj**.

Tryb jest **wolniejszy i droższy**. Poziom `effort` (`low` / `medium` / `high`,
domyślnie z `OPENAI_REASONING_EFFORT`) steruje głębokością rozumowania — dla
Gemini jest mapowany na budżet myślenia (thinking budget). Wyświetlane jest
wyłącznie oficjalne podsumowanie — nigdy surowy tok rozumowania.

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

## Automatyczny zapis wyników na dysk

W panelu bocznym (sekcja **💾 Output**) opcja **„Auto-save results to disk"**
jest **domyślnie włączona**. Każda generacja zapisywana jest do osobnego
podfolderu w katalogu wyników (domyślnie `results/`, konfigurowalny przez
`RESULTS_DIR`):

```
results/
└── political_event_chatgpt_img_20260630_084723/
    ├── political_event_chatgpt_img_20260630_084723.png    ← obraz
    ├── political_event_chatgpt_img_20260630_084723.json   ← metadane
    └── political_event_chatgpt_img_20260630_084723.txt    ← reasoning + prompt (tylko w trybie thinking)
```

- Bez trybu reasoning powstają **2 pliki** (obraz + JSON).
- Z trybem reasoning powstają **3 pliki** (dochodzi `.txt` z podsumowaniem
  rozumowania i użytym promptem).
- Wszystkie pliki mają tę samą nazwę bazową, poprzedzoną prefiksem
  **`<kategoria>_<dostawca>`**, gdzie dostawca to **`chatgpt`**, **`gemini`**
  lub **`grok`**.
- **Kategoria** jest opcjonalna — wpisujesz ją w panelu bocznym (sekcja
  💾 Output). Spacje są zamieniane na `_`. Gdy nie podasz kategorii, nazwa
  zaczyna się od dostawcy (np. `chatgpt_img_...`).
- Przy kilku obrazach w jednej generacji pliki obrazów są numerowane (`_1`, `_2`, …).

> Katalog `results/` jest w `.gitignore`, więc wyniki nie trafiają do repozytorium.

---

## Metadane przebiegu (pobieranie JSON)

Po wygenerowaniu obrazu w `image_app.py` pojawia się sekcja **📄 Run metadata**
z przyciskiem **„Download metadata (JSON)"**. Plik zawiera dane pochodzące
**z oficjalnego API** (nie ze scrapowania przeglądarki):

- prompt użytkownika i finalny prompt użyty do generacji,
- dostawca, model, parametry (rozmiar/jakość/aspect ratio/liczba obrazów),
- znaczniki czasu rozpoczęcia/zakończenia,
- dane trybu reasoning (model, effort, czas, `reasoning.summary`),
- informacje o obrazie (typ, rozmiar w bajtach, wymiary w pikselach),
- `provider_response` (m.in. `usage`/liczba tokenów, `revised_prompt`).

Plik zawiera też pole **`not_available_from_api`** — listę danych, które są
dostępne **wyłącznie przez scrapowanie interfejsu przeglądarki** i nie mają
odpowiednika w API (np. podpisany URL CDN, atrybuty DOM, surowy tok myślenia).

---

## Testy

Logika backendu generatora obrazów (`image_core.py`) jest pokryta testami,
które nie wymagają kluczy API ani połączenia z siecią:

```powershell
python -m pip install pytest
python -m pytest -q
```

---

## Najczęstsze problemy

- **`streamlit: not recognized`** – nie aktywowałeś środowiska (`venv\Scripts\activate`) lub biblioteki nie są zainstalowane. Spróbuj `python -m streamlit run app.py`.
- **`python: not recognized`** – Python nie jest zainstalowany lub nie dodano go do PATH (zainstaluj ponownie z zaznaczoną opcją „Add Python to PATH”).
- **Błąd 401 / „Invalid API key”** – zły lub pusty klucz w pliku `.env`.
- **Błąd 429 / „insufficient_quota”** (OpenAI) lub **402 / „Insufficient Balance”** (DeepSeek) – brak środków na koncie, doładuj je u dostawcy.
- **„Model not available” / „not found”** – wybrany model nie jest dostępny dla Twojego konta albo zmieniło się jego ID. W generatorze obrazów możesz wpisać własne ID w polu **Custom model**.
