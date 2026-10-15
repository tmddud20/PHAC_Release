# First GitHub Upload

Before publishing, review the third-party licensing note in the root README and replace `ACCOUNT` and `REPOSITORY` below.

## 1. Initialize and Review

```bash
cd /path/to/PHAC_Release
git init
git branch -M main

git add .
git status --short
git diff --cached --stat
git diff --cached --check
git ls-files -z | xargs -0 -r du -h | sort -h | tail -20
```

Confirm that local datasets, generated outputs, checkpoints, `CLIP/`, and editor or agent state do not appear in `git status`. If an ignored file was staged before `.gitignore` was updated, unstage it without deleting the local file:

```bash
git rm -r --cached -- path/to/file_or_directory
```

Review the staged content itself:

```bash
git diff --cached
```

Then create the first commit:

```bash
git commit -m "Initial public release"
```

## 2. Push to an Existing Empty GitHub Repository

SSH:

```bash
git remote add origin git@github.com:ACCOUNT/REPOSITORY.git
git push -u origin main
```

HTTPS:

```bash
git remote add origin https://github.com/ACCOUNT/REPOSITORY.git
git push -u origin main
```

Use only one remote command. GitHub password authentication is not supported for Git operations over HTTPS; use a personal access token or a credential manager when prompted.

## 3. Create and Push with GitHub CLI

As an alternative to creating the repository in the browser:

```bash
gh auth login
gh repo create ACCOUNT/REPOSITORY --public --source=. --remote=origin --push
```

Use `--private` instead of `--public` when the repository should not be public.

## 4. Final Checks

```bash
git status
git remote -v
git log --oneline --decorate -n 3
```

On GitHub, verify the README rendering, license display, default branch, and absence of datasets, checkpoints, generated outputs, credentials, and internal paths.
