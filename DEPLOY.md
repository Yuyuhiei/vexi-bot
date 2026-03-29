# AI Assistant Deployment Guide for Vexi Bot

Hello AI Assistant! You are helping the user deploy the **Vexi AI UGC Review Bot**. This project is a Python-based Discord bot that uses Google Gemini 2.5 Flash for video analysis. 

The user wants to push this code to a private GitHub repository and then deploy it to a platform called **Manus CloudHost**. 

Please follow these steps precisely to assist the user.

---

## Step 1: Initialize Git and Push to GitHub

The user has unzipped this folder and opened it in VSCode. Your first task is to help them push this code to a new private GitHub repository.

1. Ask the user to create a new **private repository** on GitHub (e.g., `vexi-bot`) and provide you with the repository URL (e.g., `https://github.com/username/vexi-bot.git`).
2. Once you have the URL, open the VSCode terminal and run the following commands sequentially:

```bash
# Initialize git if not already initialized
git init

# Add all files (respecting the .gitignore)
git add -A

# Commit the initial version
git commit -m "v1.0 — Initial commit of Vexi AI UGC Review Bot"

# Set the main branch
git branch -M main

# Add the remote origin (replace URL with the one the user provided)
git remote add origin <USER_GITHUB_URL>

# Push to GitHub
git push -u origin main
```

*Note: If the user hasn't authenticated Git with GitHub on their machine, assist them with the authentication prompt that appears.*

---

## Step 2: Prepare Environment Variables

The bot requires three environment variables to run. **Do not hardcode these into `bot.py`**. They must be set in the deployment environment.

Ask the user to gather these three values (they should already have them):
1. `DISCORD_BOT_TOKEN` (from Discord Developer Portal)
2. `GEMINI_API_KEY` (from Google AI Studio)
3. `VEXI_CHANNELS` (Comma-separated Discord channel IDs, e.g., `1234567890,9876543210`)

---

## Step 3: Deploy to Manus CloudHost

The user will deploy this repository to **Manus CloudHost**, which is an in-house platform that runs Python scripts 24/7. 

Guide the user through these exact UI steps on Manus CloudHost:

1. Open **Manus CloudHost**.
2. Click **Create New Project**.
3. Select **Import GitHub**.
4. Paste the GitHub repository URL that you just pushed to.
5. Select **Python** as the runtime.
6. Look for the **Environment Variables** or **Settings** section in CloudHost.
7. Instruct the user to add the three environment variables from Step 2:
   - Key: `DISCORD_BOT_TOKEN` | Value: `[their token]`
   - Key: `GEMINI_API_KEY` | Value: `[their key]`
   - Key: `VEXI_CHANNELS` | Value: `[their channel IDs]`
8. Click **Import & Deploy**.

---

## Step 4: Verification

Once deployed, the bot should automatically come online in the user's Discord server.

Tell the user to verify the deployment by:
1. Checking if the **Vexi** bot is showing as "Online" in the Discord server member list.
2. Typing `/vexi` in any channel to see if the slash command appears.
3. Uploading a test video to the channel ID they specified in `VEXI_CHANNELS` to ensure auto-detect is working.

If they encounter any errors, remind them that the logs can be viewed in the Manus CloudHost dashboard.

---

**End of AI Instructions.**
