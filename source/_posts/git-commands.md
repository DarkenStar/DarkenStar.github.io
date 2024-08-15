---
title: Git Commands
date: 2024/8/13 12:45:25
tags: common
excerpt: Here's a breakdown of the Git development workflow, starting from initialization, along with the corresponding commands
---
Here's a breakdown of the Git development workflow, starting from initialization, along with the corresponding commhexands:

# Development Flow

**1. Initialization**

* **Purpose:** Create a new Git repository to track your project's files.
* **Command:** `git init`

**2. Staging Changes**

* **Purpose:** Select which changes you want to include in your next commit.
* **Command:** `git add <file_name>` (or `git add .` to add all changes)

**3. Committing Changes**

* **Purpose:** Record a snapshot of your staged changes in the repository's history.
* **Command:** `git commit -m "Commit message"`

**4. Branching**

* **Purpose:** Create a new branch to work on a specific feature or bug fix, isolating it from the main development line.
* **Command:** `git checkout -b <branch_name>`

**5. Working on Your Branch**

* **Purpose:** Make changes to your files, test, and debug your code.
* **Commands:**
  * `git add <file_name>` (to stage changes)
  * `git commit -m "Commit message"` (to record changes)

**6. Syncing with Remote Repository (if applicable)**

* **Purpose:** Push your local changes to a remote repository (e.g., on GitHub or GitLab) for collaboration.
* **Commands:**
  * `git remote add origin <remote_url>` (to connect to the remote repository)
  * `git push origin <branch_name>` (to push your branch)

**7. Pulling Changes from Remote Repository (if applicable)**

* **Purpose:** Fetch and integrate changes from the remote repository into your local branch.
* **Command:** `git pull origin <branch_name>`

**8. Resolving Conflicts (if necessary)**

* **Purpose:** Merge changes from different branches or resolve inconsistencies between your local and remote versions.
* **Command:** `git merge <branch_name>` (and manually edit files to resolve conflicts)

**9. Merging Your Branch**

* **Purpose:** Integrate your completed feature or bug fix into the main development branch (often called `main` or `master`).
* **Command:**
  * `git checkout main` (switch to the main branch)
  * `git merge <branch_name>` (merge your branch)

**10. Pushing to Remote Repository (if applicable)**

* **Purpose:** Update the remote repository with the merged changes.
* **Command:** `git push origin main`
