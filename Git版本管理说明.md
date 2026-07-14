# Git 版本管理说明

当前版本：

- V1 固定标签：`v1-h30-real-robot`
- V1 稳定分支：`main`
- V2 开发分支：`feature/v2-state-memory-h30`
- GitHub：`git@github.com:wdqdp/myvla2.git`

## 1. 切换版本

切换前先检查是否有未保存的修改：

```bash
git status
```

切换到 V2 继续开发：

```bash
git switch feature/v2-state-memory-h30
```

查看完全固定的 V1：

```bash
git switch --detach v1-h30-real-robot
```

查看结束后返回 V2：

```bash
git switch feature/v2-state-memory-h30
```

如果需要修改 V1，应从标签新建分支：

```bash
git switch -c hotfix/v1-real-robot v1-h30-real-robot
```

## 2. 保存修改

```bash
git status
git diff
git add <修改的文件>
git diff --cached
git commit -m "feat: 简要说明本次修改"
```

常用提交前缀：`feat` 表示新功能，`fix` 表示修复，`docs` 表示文档修改。

如果修改尚未完成，但需要临时切换分支：

```bash
git stash push -u -m "临时修改说明"
git switch <目标分支>
```

恢复临时修改：

```bash
git stash pop
```

## 3. 提交到 GitHub

当前分支已经关联远端时：

```bash
git push
```

第一次推送新分支时：

```bash
git push -u origin <分支名>
```

完成一个正式版本后，可以创建并推送标签：

```bash
git tag -a <版本标签> -m "版本说明"
git push origin <版本标签>
```

提交和推送前建议始终执行 `git status`，确认没有数据集、模型权重、缓存或日志被误加入。
