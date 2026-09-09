# 有限OU幅值比不能单独识别网络误差

本轮在官方XL的OU方法对照等待期间检查既有理论中的归因。原始数值
保留，但“低于线性化degree2预期，因此必有神经半群不一致”的推断
需要撤回。无需训练即可给出精确population反例。

data-time u下，Z_u=uX+(1-u)E，X~N(0,V)，E~N(0,1)。令
c_u²=u²+(1-u)²，alpha_u=u/c_u，delta=V-1，Y=Z_u/c_u。
Y的精确方差为1+delta*alpha_u²，其相对标准高斯score为

    r_u(y) = delta*alpha_u²/(1+delta*alpha_u²) * y。

将u+h处的相对score按alpha_u/alpha_(u+h)重定时后相减，再按原
OU工具转换回velocity，得到与代码独立的闭式。分母是精确密度的
归一化效应，只有在|delta|alpha²足够小时才可忽略。

| 数据方差V | u | D(u,u+2h)/D(u,u+h) | 两方向cos |
|---|---:|---:|---:|
| 2 | .02 | 2.050112 | 1 |
| 16 | .02 | 1.899659 | 1 |
| 64 | .02 | 1.541230 | 1 |
| 64 | .05 | 1.265862 | 1 |
| 256 | .02 | .875672 | 1 |

这里h=1/32。这是同一精确Gaussian路径的两个有限查询，网络误差
严格为零。相对密度并不是只含一个Hermite模态；不能把score的简单
空间形式误当成相对密度在线性化下的单一时间幂。

脚本experiments/audit_ou_exact_gaussian_scale_ratio.py在预定方差
2/16/64/256/1024与u=.02/.05上核验全部10例，原OU实现与独立
relative-score闭式误差均<1e-11。会话48729 exit0，计算.008729 CPU秒。
结果ou_exact_gaussian_scale_ratio.json在便携结果目录。

该反例不证明真实SiT/RAE的幅值比由Gaussian covariance造成，也不
推翻先前RAE非仿射probe。它只表明当前幅值统计不能区分有限尺度的
population效应与神经模型误差，不应用其选择新的纠偏方向或强度。
官方XL raw/OU质量对照继续原样运行，未因这项检查调整方法。
