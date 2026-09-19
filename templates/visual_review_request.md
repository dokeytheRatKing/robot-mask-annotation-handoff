# 给能看图的 Codex 的小批复核请求

数据：填写 clip.json 路径、task/episode/camera、原图尺寸、source frame 范围。
任务相关对象：填写本任务确认的 object/instance 列表及 robot parts 规则。
图片：列出当前 RGB、前后帧、当前候选 overlay，提供可实际打开的本地文件。

请实际查看原图与邻帧，逐对象判断身份与可见表面。需要 seed 时给出原图像素坐标的框/正负点及 source frame，不在缩略图上猜边界。运行 SAM 后回看 seed overlay，再看非 seed 与出入画节点。模型空输出不等于 absent；不确定时保留 unknown。

输出：decision JSON、查看过的文件/帧列表、提示及坐标系、候选来源/hash、需修复的最小时间窗口、未决原因。没有人工确认则 human_confirmed=false。不要基于模型分数写 approved，不要从 object mask 扣除 robot pixels。
