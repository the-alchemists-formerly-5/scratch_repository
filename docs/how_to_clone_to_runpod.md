# Cloning and Updating a GitHub Repository on a Remote Machine via SSH

This guide explains how to clone a GitHub repository to a remote machine that you have SSH access to, and how to keep it updated.

## Prerequisites

- Git installed on your local machine
- SSH access to the remote machine
- A GitHub repository you want to clone and update

## Initial Setup

1. **Clone the GitHub repository locally**

   ```
   git clone https://github.com/username/repository.git
   cd repository
   ```

   Replace `username/repository` with the actual GitHub repository path.

2. **Add a new remote for your SSH destination**

   ```
   git remote add remote_name ssh://user@host:port/path/to/repository
   ```

   Replace:
   - `remote_name` with a name for your remote (e.g., `runpod`)
   - `user` with your username on the remote machine
   - `host` with the hostname or IP address of the remote machine
   - `port` with the SSH port (if non-standard)
   - `/path/to/repository` with the path where you want the repository on the remote machine

3. **Initialize a bare repository on the remote machine**

   SSH into your remote machine:
   ```
   ssh user@host -p port -i /path/to/ssh/key
   ```

   Once connected, create the directory and initialize a bare Git repository:
   ```
   mkdir -p /path/to/repository
   cd /path/to/repository
   git init --bare
   exit
   ```

4. **Push the repository to your remote machine**

   Back on your local machine, in your repository directory:
   ```
   GIT_SSH_COMMAND='ssh -i /path/to/ssh/key -p port' git push remote_name main
   ```

   Replace:
   - `/path/to/ssh/key` with the path to your SSH key
   - `port` with the SSH port number
   - `remote_name` with the name you chose in step 2
   - `main` with your default branch name if different

## Updating the Repository

### Pushing Latest Changes to the Remote Machine

After making changes to your local repository:

1. Commit your changes:
   ```
   git add .
   git commit -m "Description of changes"
   ```

2. Push to the remote machine:
   ```
   GIT_SSH_COMMAND='ssh -i /path/to/ssh/key -p port' git push remote_name main
   ```

### Pushing Latest Changes to GitHub

To update the original GitHub repository:

1. Ensure your local repository is up to date with the remote machine:
   ```
   GIT_SSH_COMMAND='ssh -i /path/to/ssh/key -p port' git pull remote_name main
   ```

2. Push to GitHub:
   ```
   git push origin main
   ```

### Keeping All Repositories in Sync

To ensure your local, remote, and GitHub repositories are all in sync:

1. Pull from GitHub:
   ```
   git pull origin main
   ```

2. Push to the remote machine:
   ```
   GIT_SSH_COMMAND='ssh -i /path/to/ssh/key -p port' git push remote_name main
   ```

3. Push to GitHub (if you made any local changes):
   ```
   git push origin main
   ```

## Troubleshooting

If you encounter issues:

1. Verify SSH access:
   ```
   ssh user@host -p port -i /path/to/ssh/key
   ```

2. Check permissions on the remote machine:
   ```
   ls -la /path/to/repository
   ```

3. Ensure you have write permissions in the destination directory on the remote machine.

## Note

Always make sure to pull the latest changes before pushing to avoid conflicts. If you're working with multiple branches, replace `main` with the appropriate branch name in the commands above.