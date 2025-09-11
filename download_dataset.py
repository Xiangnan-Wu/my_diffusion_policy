from huggingface_hub import snapshot_download

# 替换成你要下载的数据集 ID
repo_id = "XiangnanW/continuous_data_dev"

# 下载整个数据集到本地 datasets/food101
local_path = snapshot_download(repo_id, repo_type="dataset", local_dir="/home/wxn/Projects/diffusion_policy/data/continuous_data_dev")

print("数据集下载到:", local_path)