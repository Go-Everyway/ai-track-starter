# deploy

The function you ship in Module 0: one fixed question to a model, answered as JSON at a public URL.

- `app.py` holds the function. Vercel loads the Flask `app` from it.
- `requirements.txt` lists Flask, which is the only dependency.
- `test_app.py` holds local checks. They need no network and no key: `python deploy/test_app.py`.

## Ship it on Vercel

These steps were checked against Vercel's docs on 2026-11-11. Steps change, so if something doesn't match, follow the [Python runtime docs](https://vercel.com/docs/functions/runtimes/python).

```bash
npm i -g vercel                     # the Vercel CLI (needs Node.js; --environment needs CLI 64.7.0+)
vercel login
cd deploy
vercel                              # creates the project; a new project's first deploy is production
vercel env add LLM_API_KEY --environment production --sensitive
vercel env add LLM_MODEL --environment production
vercel --prod                       # redeploy: env var changes only apply to new deployments
curl https://<your-project>.vercel.app/
```

Use the short production URL (`<your-project>.vercel.app`). Depending on your project's Deployment Protection setting, the longer per-deployment URLs may ask for a Vercel login.

## Before you share the URL

Anyone with the URL can call it, and every call spends your money. Two things keep that small:

- The prompt is fixed and replies are capped at 60 tokens. Don't add a parameter that lets the caller change either.
- Set a credit limit on the key at your provider. That's the cap that holds even if the code is wrong.

When you're done, delete the project or remove the key from it: `vercel env remove LLM_API_KEY --environment production`.
