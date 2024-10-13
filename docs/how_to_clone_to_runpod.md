# Cloning scratch_repository Locally and Pushing to Remote Machine via SSH

Follow these steps to clone the scratch_repository from GitHub to your local machine and then transfer it to your remote machine using SSH.

## Steps

1. **Clone the GitHub repository locally**

   On your local machine:
   ```
   git clone git@github.com:the-alchemists-formerly-5/scratch_repository.git
   cd scratch_repository
   ```

2. **Create a tar archive of the repository**

   Still on your local machine:
   ```
   cd ..
   tar -czf scratch_repository.tar.gz scratch_repository
   ```

3. **Transfer the archive to the remote machine using SCP**

   ```
   scp -P 18952 -i ~/.ssh/id_rsa_runpod_ghiret scratch_repository.tar.gz root@213.173.108.216:/workspace/
   ```

4. **SSH into the remote machine and extract the archive**

   ```
   ssh root@213.173.108.216 -p 18952 -i ~/.ssh/id_rsa_runpod_ghiret
   ```

   Once connected:
   ```
   cd /workspace
   tar -xzf scratch_repository.tar.gz --overwrite
   rm scratch_repository.tar.gz
   ```

5. **Set up Git configuration on the remote machine (if needed)**

   Still on the remote machine:
   ```
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

   Replace "Your Name" and "your.email@example.com" with your actual details.

6. **Exit the SSH session**

   ```
   exit
   ```

## Updating the Repository

### Pulling Latest Changes on the Remote Machine

1. SSH into your remote machine:
   ```
   ssh root@213.173.108.216 -p 18952 -i ~/.ssh/id_rsa_runpod_ghiret
   ```

2. Navigate to your repository:
   ```
   cd /workspace/scratch_repository
   ```

3. Pull the latest changes:
   ```
   git pull origin main
   ```

4. Exit the SSH session:
   ```
   exit
   ```

### Pushing Changes from the Remote Machine to Your Local Machine

If you've made changes on the remote machine and want to update your local repository:

1. On your local machine:
   ```
   cd path/to/local/scratch_repository
   git remote add runpod ssh://root@213.173.108.216:18952/workspace/scratch_repository
   git pull runpod main
   ```

2. After pulling, you can push these changes to GitHub if desired:
   ```
   git push origin main
   ```

## Note

This method ensures that your GitHub credentials are never entered on the remote machine, enhancing security. Always be cautious when working with repositories on shared or public machines.
