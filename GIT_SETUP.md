# Git Setup for Personal GitHub Account

This guide shows how to use your personal GitHub account for this repository without affecting your corporate Git configuration.

## ✅ Option 1: Repository-Specific Configuration (Recommended)

Set Git credentials for **this repository only**:

```bash
cd /Users/ndi76jc/Desktop/development/RL

# Set your personal name and email for this repo
git config user.name "Your Personal Name"
git config user.email "your.personal@email.com"

# Verify the settings
git config user.name
git config user.email
```

This won't affect your global (corporate) Git settings!

## ✅ Option 2: Conditional Git Config (Advanced)

Edit your global Git config to automatically use different credentials based on directory:

```bash
# Edit your global Git config
nano ~/.gitconfig
```

Add this section:

```ini
# Your existing corporate config
[user]
    name = Your Corporate Name
    email = your.corporate@company.com

# Conditional config for personal projects
[includeIf "gitdir:~/Desktop/development/"]
    path = ~/.gitconfig-personal
```

Then create `~/.gitconfig-personal`:

```bash
nano ~/.gitconfig-personal
```

Add:

```ini
[user]
    name = Your Personal Name
    email = your.personal@email.com
```

Now all repos under `~/Desktop/development/` will use your personal account!

## 🔑 SSH Key Setup (Recommended for Multiple Accounts)

### Step 1: Generate a Personal SSH Key

```bash
# Generate a new SSH key for your personal GitHub
ssh-keygen -t ed25519 -C "your.personal@email.com" -f ~/.ssh/id_ed25519_personal

# Start the SSH agent
eval "$(ssh-agent -s)"

# Add the key to the agent
ssh-add ~/.ssh/id_ed25519_personal
```

### Step 2: Add SSH Key to GitHub

```bash
# Copy the public key
cat ~/.ssh/id_ed25519_personal.pub
```

1. Go to GitHub.com (personal account)
2. Settings → SSH and GPG keys → New SSH key
3. Paste the key and save

### Step 3: Configure SSH for Multiple Accounts

Edit `~/.ssh/config`:

```bash
nano ~/.ssh/config
```

Add:

```
# Corporate GitHub
Host github.com-corporate
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519  # Your corporate key

# Personal GitHub
Host github.com-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_personal
```

### Step 4: Use the Personal Account for This Repo

When adding the remote, use the personal host:

```bash
# Instead of: git remote add origin git@github.com:username/repo.git
# Use:
git remote add origin git@github.com-personal:your-personal-username/RL-DQN.git
```

## 📝 Quick Setup Commands

Here's a complete setup for this repository:

```bash
cd /Users/ndi76jc/Desktop/development/RL

# 1. Set personal credentials for this repo
git config user.name "Your Personal Name"
git config user.email "your.personal@email.com"

# 2. Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Training outputs
*.pth
*.pkl
*.png
*.jpg

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
EOF

# 3. Add all files
git add .

# 4. Initial commit
git commit -m "Initial commit: DQN implementation for MountainCar and Pong"

# 5. Create GitHub repo (do this on github.com first)
# Then add remote (replace with your username):
git remote add origin git@github.com-personal:YOUR_USERNAME/RL-DQN.git

# Or if using HTTPS:
git remote add origin https://github.com/YOUR_USERNAME/RL-DQN.git

# 6. Push to GitHub
git branch -M main
git push -u origin main
```

## 🔍 Verify Your Configuration

Check which account will be used:

```bash
# Check local (repo-specific) config
git config user.name
git config user.email

# Check global (corporate) config
git config --global user.name
git config --global user.email

# See all configs
git config --list --show-origin
```

## 🚨 Troubleshooting

### Issue: Still using corporate account

```bash
# Make sure you're in the repo directory
cd /Users/ndi76jc/Desktop/development/RL

# Set local config (overrides global)
git config user.name "Personal Name"
git config user.email "personal@email.com"

# Verify
git config user.email  # Should show personal email
```

### Issue: SSH authentication fails

```bash
# Test SSH connection
ssh -T git@github.com-personal

# Should see: "Hi YOUR_USERNAME! You've successfully authenticated..."
```

### Issue: Wrong account used in commit

```bash
# Amend the last commit with correct author
git commit --amend --author="Personal Name <personal@email.com>"
```

## 📋 Summary

**For this repo only (easiest):**
```bash
git config user.name "Your Personal Name"
git config user.email "your.personal@email.com"
```

**For all personal projects (better):**
- Use conditional includes in `~/.gitconfig`
- Set up separate SSH keys

**Your corporate Git setup remains untouched!** ✅

## 🔗 Useful Commands

```bash
# Check current config
git config --list

# Check where config comes from
git config --list --show-origin

# Remove local config (fall back to global)
git config --unset user.name
git config --unset user.email

# Check remote URL
git remote -v

# Change remote URL
git remote set-url origin NEW_URL
```

## 📚 Additional Resources

- [GitHub: Managing multiple accounts](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-your-personal-account/managing-multiple-accounts)
- [Git Config Documentation](https://git-scm.com/docs/git-config)
- [SSH Key Setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
