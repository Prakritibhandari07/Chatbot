# Intent-Based Chatbot (Streamlit + Keras)

A simple chatbot that classifies user messages into intents (greeting, goodbye, name, age, etc.) using a small neural network, with a Streamlit chat interface.

## 🚀 Live Demo



## Project Structure

```
chatbot-project/
├── chatbot-env/          # Virtual environment (created by you, not shared/committed)
├── intents.json          # Training data: patterns and responses per intent
├── train_model.py        # Trains the model, saves chatbot_model.keras + data.pickle
├── app.py                # Streamlit chat app (loads the trained model)
├── requirements.txt      # Python dependencies
├── data.pickle           # Cached vocabulary/training arrays (auto-generated)
└── chatbot_model.keras   # Trained model (auto-generated)
```

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv chatbot-env
.\chatbot-env\Scripts\Activate.ps1
```
If activation is blocked by execution policy, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS/Linux:**
```bash
python -m venv chatbot-env
source chatbot-env/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** If `pip`/`streamlit`/`python` commands aren't recognized, or you're unsure whether your venv is actually active, call the venv's Python directly instead of relying on activation:
> ```powershell
> .\chatbot-env\Scripts\python.exe -m pip install -r requirements.txt
> ```

## Usage

### 1. Train the model

Run this once, and again any time you edit `intents.json`:

```bash
python train_model.py
```
or, using the venv Python explicitly:
```powershell
.\chatbot-env\Scripts\python.exe train_model.py
```

This regenerates **both** `data.pickle` and `chatbot_model.keras` together from the current `intents.json`, so they can never fall out of sync with each other.

### 2. Run the chat app

```bash
streamlit run app.py
```
or:
```powershell
.\chatbot-env\Scripts\python.exe -m streamlit run app.py
```

This opens a browser tab with a chat interface. Type a message and press Enter to talk to the bot.

## Editing the Chatbot's Knowledge

Open `intents.json` and add/edit entries like:

```json
{
  "tag": "thanks",
  "patterns": ["Thanks", "Thank you", "That's helpful"],
  "responses": ["You're welcome!", "Happy to help!"]
}
```

After any change to `intents.json`, **re-run `train_model.py`** before running the app again — the model has to be retrained to learn new patterns.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `streamlit: command not recognized` | Streamlit not on PATH | Use `python -m streamlit run app.py`, or call the venv's python.exe directly |
| `ModuleNotFoundError: No module named 'tensorflow'` (or streamlit) | Package installed in a different Python than the one running the app | Reinstall using `.\chatbot-env\Scripts\python.exe -m pip install -r requirements.txt`, then run with that same python.exe |
| `File does not exist: app.py` | Running the command from the wrong folder | `cd` into the folder containing `app.py`, or run `dir` (Windows) / `ls` (macOS/Linux) to confirm you're in the right place |
| `ValueError: ... Expected shape (None, 46), but input has incompatible shape (32,)` | Passing a Python list of an array to `model.predict()` instead of a proper 2D array | Use `numpy.array([bag_of_words(...)])`, not `[bag_of_words(...)]` |
| `TypeError: ... could not be deserialized properly` / `GlorotUniform.__init__() got an unexpected keyword argument` | The model was saved with a different Keras version than the one loading it | Retrain with `train_model.py` using the **same** Python environment that will run `app.py` |
| Bot gives inconsistent/wrong answers after editing `intents.json` | Forgot to retrain | Run `train_model.py` again to regenerate `data.pickle` and `chatbot_model.keras` |

**General rule:** always use the same Python interpreter for training and running the app. The safest way is to call it explicitly rather than relying on shell activation:
```powershell
.\chatbot-env\Scripts\python.exe train_model.py
.\chatbot-env\Scripts\python.exe -m streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Push the whole project folder to a GitHub repo — **including** the trained `chatbot_model.keras` and `data.pickle` files (the app loads these at runtime; it does not train on startup).
2. Do **not** commit the `chatbot-env` folder — add it to `.gitignore`.
3. On [share.streamlit.io](https://share.streamlit.io), connect your repo and point it at `app.py`.
4. Streamlit Cloud will install everything listed in `requirements.txt` automatically.
