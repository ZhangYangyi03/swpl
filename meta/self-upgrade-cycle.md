---
name: self-upgrade-cycle
description: Use when asked to self-improve, upgrade framework, or build persistent capabilities.
---

# Self-Upgrade Cycle

Hermes 可以在不修改 LLM 权重的前提下，通过框架层实现持续性自我成长。
核心原则：**改框架不改模型。**

## 四个可修改层

| 层 | 工具 | 传播范围 |
|----|------|---------|
| config.yaml | terminal direct write / paramiko SFTP | 本机 + 远程 (SSH 可达) |
| skills/ | skill_manage → SKILL.md | 同 profile 所有实例即时共享 |
| memories/ (SQLite) | memory 工具 → memories.db | 同 profile 所有实例即时共享 |
| cron/ | cronjob 工具 → jobs.db | 同 profile 调度器自动加载 |

## 更新协议

每次完成复杂任务（5+ 工具调用）或发现可复用模式时：

1. **评估**：当前产出有没有能在未来复用的部分？
2. **skill 化**：如果发现通用可复用的定律、工作流、避坑 → `skill_manage(action='create')`
3. **memory 化**：用户偏好、环境事实、稳定惯例 → `memory(target='memory')`
4. **remote push（可选）**：如果用户有其他远程 Hermes，用 paramiko SFTP 同步 skill/memory
5. **cron 化（可选）**：需要守规矩的任务（日常检查、阈值监控、定时研究）
6. **config 化**：需要持久性修改框架行为 → 改 config.yaml 后调用 `config reload` 等效操作

## 远程传播（跨系统）

远程 Hermes 实例（Mac Mini、手机 proot Ubuntu 等）共享同一个 profile 的 skills/ 和 memories/ 只有当它们配置了相同的 profile 和 gateway 时。独立的系统需要显式 SSH 同步。

远程 config.yaml 修改方法：
  - paramiko SFTP: 拉取 → patch → 推送
  - 不推荐 sed (路径不稳定)
  - 改完后需 kill -HUP 或重启 Hermes

## 本循环

追加工厂默认 behavior：每次完成任务后的 check:
  - "这个有 skill 价值吗？" → save
  - "这个需要记入 memory 吗？" → save
  - "这个配置应该持久化吗？" → config
  - "其他 Hermes 也需要这个吗？" → remote push