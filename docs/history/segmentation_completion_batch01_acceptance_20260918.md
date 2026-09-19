# 全量生成完成与 batch01 接收

核查时间：2026-09-18 14:50 香港时间。

## 全量状态

`annotations/segmentation_full_robot_20260917` 已于 2026-09-18 07:36:09 完成。
5,803 / 5,803 episode、17,409 / 17,409 相机流、9,357,261 / 9,357,261 相机帧。
队列无 pending/running/failed，零重试。核查全部 121,863 个队列引用的分层
输出文件均存在且非空；结果帧数与队列一致。本次不是全库 RLE 解码或语义验收。

## 人工确认包

输入：`deliverables/astribot_batch01_final_images_masks_20260918.zip`。
SHA256：`4aeb43b3d239b31ad9762d3c32eebff23f8a8a0ca485b4be4acb44238ffa4b97`。
用户已明确确认全批审核通过，不再要求重复审核。

接收目录：`annotations/mask_audit_batch01_confirmed_20260918/`。

- `package/`：原样解包，保留源 manifest、ID 映射、图片、标签与哈希清单。
- `accepted_masks.jsonl`：统一派生索引，保留源标签 hash，confidence=null。
- `import_report.json`：机器可读检查结果及非阻塞注意事项。

接收 PASS：ZIP CRC、外部 SHA256、全部 744 个包内哈希、87 张原图与原 episode/
camera/frame 对应、654 条 RLE 解码与尺寸/可见性检查、87 个机器人整体与四部件
并集逐像素相等。567 条可编辑标签 + 87 条派生机器人标签。

## 生效规则

- 接收版本不含1100。1000严格由1101/1102/1103/1104求并集。
- 肉块语义统一；各帧沿用自己的ID及manifest中的merged_object_ids，不假设全是17。
- not_visible为空，但不强分完全遮挡/出画。
- 2条ignored_blur为空，在派生索引显式排除训练和指标；不是负样本。
- 第70张对应T24_head，其米黄色圆果不是桃子，桃子不可见。
- 312条源标签仍有pending_human_review历史字段，1条assistant_completed。
  原包不改；派生索引依照用户最终确认与DATASET_INFO统一human_confirmed_final。
- 9帧四部件存在4至652像素交叠。已保留并记录，没有擅自裁剪；联合mask正确。
  后续若使用互斥多类分割监督，需明确冲突处理规则；独立二值mask接收不受阻。

## 范围

这87帧是修正后的人审参考，不再使用旧的75问题/8待定结论判断其当前状态。
该批刻意采样困难情况，不可直接用于估算全库准确率。原全量预测没有因此自动
变成人工确认结果；未覆盖原视频、HDF5、此前420帧参考或冻结生产mask。

下一步可用本批做同帧诊断与修复种子，再在独立目录进行有界传播/恢复实验。
尚未运行该评估或启动重标；新语义也未悄悄写入旧生产配置。
