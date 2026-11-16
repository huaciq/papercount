# 分支协作开发

`main` 保存主线版本，`zxy` 和 `zys` 分别用于两位开发者的日常开发。
`zys` 从包含瓦楞纸改进和第一轮材料策略迁移的 `zxy` 提交创建，创建时两者代码一致；之后各自的提交独立保存。

## 同学电脑首次获取

```powershell
git clone --branch zys https://github.com/huaciq/papercount.git
cd papercount
git branch -a
```

克隆会获取远程分支信息，工作区直接处于 `zys`，并跟踪 `origin/zys`。
`git branch -a` 可查看 `origin/main`、`origin/zxy`、`origin/zys`；无需分别克隆三次。
如需查看其他分支，先提交当前改动，再使用 `git switch main` 或 `git switch zxy`。
私有仓库需要先给同学的 GitHub 账号配置访问权限。

如果同学已经克隆过仓库，且还没有本地 `zys`：

```powershell
git fetch origin
git switch --track origin/zys
```

## Python 环境

```powershell
conda activate pytorch
cd paper_counter
python -m pip install -e .
python -m pip install pytest
python -m pytest tests -q
```

如果本机没有 `pytorch` 环境，可先运行 `conda create -n pytorch python=3.10` 创建，再执行上述命令。
运行方式见项目 [README](../README.md)。
代码、配置、测试和文档通过 Git 同步；本机的新增数据集与推理输出需另行共享。
`infer.py` 顶部的输入输出路径需要按各自电脑调整。
材料策略与配方调用见 [PIPELINES.md](PIPELINES.md)。

## 日常提交

同学在 `zys` 上开发，原电脑在 `zxy` 上开发。开始前检查分支并获取本分支更新：

```powershell
git branch --show-current
git pull --ff-only
```

完成一项修改并验证后，明确选择本次涉及的文件，用 `git add <文件路径>` 暂存，再提交和推送：

```powershell
git diff --cached
git commit -m "说明本次修改"
git push
```

分支不会自动继承对方后续提交。需要整合时，先获取远程最新状态，再通过 Pull Request 审查并合并到目标分支；
合并后运行相应测试及真实图片回归，确认后再更新主线。
