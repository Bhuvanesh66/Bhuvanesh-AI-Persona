# Deployment Notes

## Backend model configuration

This project uses a Python FastAPI backend for the chat agent and a separate Next.js frontend.

The frontend should point to a live backend via `NEXT_PUBLIC_API_URL`.

### Important environment variables for the backend

- `GROQ_API_KEY`: only required if you want to use Groq.
- `ENABLE_GROQ=false`: default, so the backend uses GitHub models only.
- `CHAT_MODEL`: should use a supported GitHub model when in GitHub-only mode.
- `VOICE_MODEL`: should use a supported GitHub voice model when in GitHub-only mode.

### Supported defaults

If your deployed backend still has deprecated model names, the code now remaps them automatically:

- `llama3-groq-70b-8192-tool-use-preview` → `llama-3.3-70b-versatile`
- `llama3-groq-8b-8192-tool-use-preview` → `llama-3.1-8b-instant`

### Recommended configuration

Use these exact values in the deployed backend `.env` or environment settings:

```env
GROQ_API_KEY=your_groq_api_key
CHAT_MODEL=llama-3.3-70b-versatile
VOICE_MODEL=llama-3.1-8b-instant
```

### Verify the backend

After deployment, verify the backend before using the frontend:

1. Visit `http://<backend-url>/health`
2. Check the returned JSON includes `model` and `use_groq`
3. Call `POST http://<backend-url>/chat/stream` or use the `/debug` endpoint to confirm Groq connectivity

### Notes

- The frontend itself does not host the backend on Vercel. It requires a separate always-on FastAPI backend.
- If the backend returns a 400 error mentioning `model_decommissioned`, update `CHAT_MODEL` and `VOICE_MODEL` to supported Groq names.
