# 时间修订的时钟项检查

这是对上一轮精确 Gaussian 幅度反例的进一步区分，不是已验证的新生成方法。
仓库 `PFR_MECHANISM_AUDIT_20260903_ZH.md` 已测试不同信息时钟并发现不等价；
本次讨论的是速度场的时间重参数化，和只换 query horizon 不同。

任意光滑 ODE dz/dt=v(t,z)，若单调时钟 phi 固定两端，则

    b(t,z)=phi'(t)*v(phi(t),z)

具有相同终点映射。令 phi=t-epsilon(t)，展开得

    b=v-epsilon*partial_t v-epsilon'*v+O(epsilon^2).

raw time-only 修订 gamma*(v(t)-v(t+h*g(t))) 只在一阶提供
-gamma*h*g*partial_t v。若只把它称为“时钟移动”，就漏掉了
-gamma*h*g'*v。这里的偏导固定 z，不是沿轨迹导数。

CPU 检查使用精确 Gaussian 场、gamma=.35、g=sin²(2*pi*t)（t<.5），
其后为零，使时钟固定端点；这是一个新的平滑控制，不是原先硬窗口的完整复现。
比较 raw、raw-gamma*h*g'*v、精确 phi'v(phi)。目标方差 V=4 时：

| h | raw 的 log 标准差偏移 | 补偿后偏移 | 精确时钟偏移 |
|---|---:|---:|---:|
| 1/32 | -.01408037 | .00031810 | <1e-16 |
| 1/64 | -.00712517 | .00007407 | <1e-16 |
| 1/128 | -.00358181 | .00001781 | <1e-16 |

V=.25、1、16 也已计算；一阶与二阶尺度符合展开，精确时钟积分残差
全部小于 1e-16。脚本 `experiments/audit_pfr_clock_compensator.py`，CSV
`experiments/results/terminal_defect_20260908/pfr_clock_compensator.csv`。

**适用边界。** 真实 PFR 主场 G 与弱头 W 不同。补偿后的增量一阶为
-partial_t(epsilon*W)，可分为

    -partial_t(epsilon*G) - partial_t(epsilon*(W-G)).

第一项是 G 的一阶时钟变换，第二项仍是实际改变终点的相对场干预。
所以即使补偿也没有一般的终点不变保证，更没有质量保证。原生 IG 的开关、
denominator floor 和硬窗口还需分段处理，不能在事件处直接应用光滑展开。

这使后续问题更具体：共享时间变化中有多少可解释为时钟效应，以及去掉这部分
后剩余相对场变化是否有益。已有 SiT strong/weak 共线分解不是这个导数分解，
但其“共享部分有用”的结果意味着不能预先认定去掉共享变化就更好。
目前不据此直接启动 5K，也不把这个基础 ODE 身份作为创新。
