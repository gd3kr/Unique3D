from huggingface_hub import snapshot_download
import os
import shutil

def download_repo(repo_id, local_dir):
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type='space',
            allow_patterns=['ckpt/*']
        )
        print(f'Repository downloaded to: {downloaded_path}')
        
        # Move ckpt folder one level up
        src = os.path.join(local_dir, 'ckpt')
        dst = os.path.join(os.path.dirname(local_dir), 'Unique3D/ckpt')
        shutil.move(src, dst)
        print(f'Moved ckpt folder to: {dst}')
        
        # Remove the downloaded_repo folder
        shutil.rmtree(local_dir)
        print(f'Removed {local_dir}')
    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == "__main__":
    repo_id = 'Wuvin/Unique3D'
    local_dir = '/root/downloaded_repo'
    download_repo(repo_id, local_dir)