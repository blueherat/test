# RAEv2同场校准：检验异步往返是否足以造成退化

原生IG常规采样始终保持不变。已有两场校准G=-drift，S=-v_full，R=2S-G，
q=z+hG(z,t)，z_new=q-hR(q,t-H)。本轮只将校准参考换成R=G，保留实际
查询位置、时间、算术顺序、所有事件和迭代数。普通采样仍为原生IG1.78。
该对照消除两场校准中额外的未来IG增量，不消除G本身包含的Full/Base组合。
所以它检验的是“校准两侧使用不同场是否必要”，不是“Full/Base不再起作用”。

固定h=.025,H=.125，Euler100官方shift8，事件0/54/83、次数2/2/1。
同一seed202609413、连续B4噪声与labels0..999、BF16/TF32、1000样本。
每样本仍110次Full调用；与既有native100 FID38.264239、两场async48.972310
配对比较。不要比较1K与历史5K的绝对FID。预计 .25 GPU小时采样，模型加载
与评估另外计时。无训练，无h/H/迭代数搜索，不因本轮结果自动扩展5K。

质量采样之前：8张original/all逐像素重现旧async，8张common/none逐像素
重现native100；common/all固定8张smoke检查有限输出、调用数和来源。
所有组保存源代码、checkpoint、输入与像素hash。正式1K官方FID完成后，用
已缓存特征独立FP64 Gram计算验证。一次1K用于定位机制，不作为新方法成功。

## 完成结果

固定1K：native100 FID38.264239，原两场async48.972310，同场50.002931。
同场IS70.16661，采样881.442秒（.24485 GPU小时），每样本110 Full调用。
原生与原校准8图逐像素重现、1K输入/checkpoint/helper来源及像素hash一致性
通过；独立FP64特征Gram FID50.002948，误差1.70e-5。驱动与审计均exit0。

结论：在该固定RAE设置下，两侧不同参考场不是退化的必要条件。同场往返
本身足以造成显著退化；但G仍含原生IG，不能说Full/Base与此完全无关。
未来IG-only组仍在运行，尚不对另一分量或完整轨迹交互作结论。
本结果为1K机制对照，不是5K或新方法；不扩展该退化组。
CSV experiments/results/terminal_defect_20260908/raev2_common_field_calibration.csv。

后续互补组已完成：future_ig FID41.881665，也未超过基线；详见互补协议。
