# Deployment Notes

## Backend model configuration

This project uses a Python FastAPI backend for the chat agent and a separate Next.js frontend.

The frontend should point to a live backend via `NEXT_PUBLIC_API_URL`.

### GitHub Models only

The backend uses **GitHub Models exclusively** with your GitHub token.

### Important environment variables for the backend

- `GITHUB_TOKEN`: required — uses your GitHub token with `Models: Read` permission.
- `CHAT_MODEL`: defaults to `gpt-4o-mini`.
- `VOICE_MODEL`: defaults to `gpt-4o-mini`.

### Recommended configuration

Use these exact values in the deployed backend `.env` or environment settings:

```env
GITHUB_TOKEN=ghp_xxxxx_your_token
CHAT_MODEL=gpt-4o-mini
VOICE_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=github
EMBED_MODEL=text-embedding-3-large
```

### Verify the backend

After deployment, verify the backend before using the frontend:

1. Visit `http://<backend-url>/health`
2. Check the returned JSON includes `model` and `use_groq`
3. Call `POST http://<backend-url>/chat/stream` or use the `/debug` endpoint to confirm Groq connectivity

### Notes

- The frontend itself does not host the backend on Vercel. It requires a separate always-on FastAPI backend.
- If the backend returns a 400 error mentioning `model_decommissioned`, update `CHAT_MODEL` and `VOICE_MODEL` to supported Groq names.
