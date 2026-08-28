# DiT B/C 事件富集确认方案（标签可靠性修订 v3）

> 2026-08-28 更新：本文件以下 v3 B/C 方案保留为不可变历史，但在任何真实
> event-screen 样本产生前已由 corrected scientific v4.1 明确取代，不能再执行。第一版
> v4 因继承了一句 candidate-C 旧 rubric 文字且 zero-screen 审计漏掉新 launcher 前缀，
> 在任何下游锁或采样前也已被 v4.1 取代并只保留作审计。当前方案改为同一
> 六类别模糊风险评估总体上的 `B_persistence` / `E_blur_gated_running_max_log`
> co-primary 检验，C 退出正式家族。详见
> [DIT_EVENT_RICH_SCIENTIFIC_V4_ZH.md](DIT_EVENT_RICH_SCIENTIFIC_V4_ZH.md)。
> FID/Inception/DINO/视觉标签只能承担外部评估和评估总体构造；它们绝不进入 B/E、
> 无标签阈值、触发或干预。

## 结论

第三池的主要问题不是已经证明 B/C 无效，而是 1,800 张里最终只有 6 张
`clear-bad`、其中 4 张属于 blur/soft-fusion。这样的阳性数不足以打开冻结的分数—标签
连接。下一轮不应继续在相同三个类别上盲目扩容，也不应把 Inception、DINO 或 FID
变成方法。更有效的设计是：

1. 用完全相同的标准采样器，只生成 endpoint，在较宽的预冻结 ImageNet 类别表上做
   小样本发生率筛查；
2. 人工标签锁定后，分别为 B（blur/fusion）和 C（所有 clear-bad）选择高风险类别；
3. 用第二组 endpoint-only 新种子验证这些类别的事件发生率；
4. 只有独立 anchor 通过，才为对应候选在第三组新种子上保存完整轨迹；
5. B、C 的公式、方向、时间窗、主 endpoint 和统计门槛全部不改。

第一阶段只决定“去哪里找更常发生的事件”，第二阶段才回答“内部轨迹量能不能在事件
完成前区分 bad 和 good”。因此 endpoint 筛查是实验抽样设计，不是论文方法。

这里不是一轮完全未受历史信息影响的 replication：B/C 和失败类型来自旧池探索。真正
前瞻独立的是新 discovery、anchor 与 confirmation 的逐行样本；第三池逐行图片、标签、
轨迹、候选分数与 embedding 均不得进入新类别排名或功效外推。

第三池还暴露了更靠前的问题：三名 reviewer 的 clear-bad binary Fleiss κ 只有 `0.147`；
29 张原始多数 bad 被单名 adjudicator 降级了 23 张，且 23 张全是 class 207；独立视觉
复核至少发现 7 张明显 bad 被漏掉。因此 v3 的首要门不是候选分数，而是先证明标签系统
能稳定识别“明显低于模型同类平均”的失败。

## 标签系统先过资格门

v3 冻结了 20 张跨类别可见教学 anchor，覆盖 clean、mild、clear blur/fusion 和 clear
topology，并为每张冻结文字理由。最重要的规则是：**能认出主体不等于质量正常**。当整张
图明显软糊、面部/物体细节被抹掉，且低于冻结模型同类的普通水平时，即使仍看得出是狗、
人或滑雪场，也必须至少标 severity 2。

这些可见 anchor 只用于教学，不进入 qualification 统计。正式 reviewer 在看到任何新
screen 图片前，必须独立完成另一套隐藏的 60 张平衡 qualification：15 clean、15 mild、
15 clear blur/fusion、15 clear topology。隐藏 gold 由两名独立专家建立，分歧交第三人，
图片、标签和理由先哈希冻结。

三名 reviewer 的每一对都必须同时满足：

- positive agreement ≥ `0.60`；
- clear-bad binary Cohen κ ≥ `0.50`；
- 每名 reviewer 对隐藏 gold 的 clear-bad recall ≥ `0.80`，non-clear specificity ≥ `0.80`。

任一门失败就 STOP，不得发放正式 screen。失败 reviewer 可更换或重新培训，但整个新 panel
必须在一套预先哈希、与失败题目不重叠的 reserve form 上重新资格测试，不能反复刷同一套题。

## 双裁决与漏报审计

screen、anchor 和 confirmation 都使用三名独立 reviewer，之后由两名已资格化、彼此独立
的 adjudicator 处理盲化审计包。单个人不能改变最终标签：

- 三名 reviewer 一致 clear-bad 的样本不可降级；
- 2-of-3 clear-bad 只有两名 adjudicator 都独立判为非 bad 才能降级，否则保留；
- 任何 reviewer 给 severity≥2、但未形成多数 bad 的图片全部进入盲复核；两名 adjudicator
  都判 clear-bad 才提升；
- 另从无人给 severity≥2 的多数阴性中，用冻结随机规则抽取等量 decoy 混入，估计漏报；
- adjudicator 看不到触发来源、票数、reviewer 身份、候选分数、embedding 或彼此结论。

这样既阻止再次出现“一名裁决者把一个类别的多数 bad 大量抹掉”，也能观察原先
downgrade-only 流程完全看不到的 false negative。抽样绝不能使用 B/C、Inception、DINO、
FID 或其他模型分数。

## 两级 endpoint-only 筛查

冻结类别表包含 84 类、7 个形态层：多关节/长毛哺乳动物、细腿/翅膀、复杂多部件
动物、人与服装交互、细杆刚性拓扑、多部件车辆/物体、柔软聚簇有机体。它排除了旧的
207、602、795 三类。

### Discovery rank

- seeds `1000..1011`，每类 12 张，共 1,008 张；
- 三名 reviewer 只看 256×256 endpoint，使用与第三池相同的模型相对严重度和 phenotype
  规则；
- B 按 `blur/soft-fusion clear-bad` 数量降序选 6 类；
- C 按所有 `clear-bad` 数量降序选 6 类；
- 并列时只用冻结的类别顺序，不看任何内部量或外部 embedding。

84 类是按形态风险有意富集的固定设计，不是 ImageNet 的随机样本，也不支持总体发生率
估计。v3 不再使用会随 batch/order 改变结果的 6-class chunk。每张图都是 singleton，随机
单元严格定义为 `(global_seed,class_id)`：用 domain
`eqvae.dit.event-rich.endpoint.v1` 和这两个十进制整数做带 NUL 分隔的 SHA-256，取 digest
前 8 bytes 的 big-endian 无符号整数并清除最高位，得到 63-bit `pair_seed`。模型和 VAE
加载完后、singleton 初始 latent 前才调用 `torch.manual_seed(pair_seed)`。

CFG 仍把 B=1 latent 复制成有序 2B conditional/null batch；250 个 ancestral DDPM step
每步都生成完整 2B `randn_like`，包括 t=0 先消耗随机数再乘零。不同 class 不共享 initial
latent 或 transition innovation，因此结果对 batch、类别顺序、GPU shard 和断点恢复不变。
它保留第三池的模型/250步/full-2B-draw 语义，但不声称复现旧三类 batch 的相关结构。

### Independent anchor

- seeds `1012..1035`，只生成前一步选中类的 endpoint；
- B/C 各 6 类，若完全不重叠最多 12 类、288 张；
- anchor 与 discovery rank 的种子完全独立；
- B 至少需要 6 个 blur/fusion bad，且至少来自 3 类；
- C 至少需要 11 个 clear-bad，且至少来自 3 类；
- 两者各自还要求至少 60 个 clean-good，保证 AUC 有对照；
- B、C 分别 GO/STOP，一个失败不阻止另一个继续。

Discovery 加 anchor 最多 1,296 张 endpoint。它们不保存轨迹，所以相比直接再做一大池
完整 trace，存储和后处理成本显著更低。

## 为什么确认池定为每类 128 张

每个候选最多 6 类、每类 128 个新确认种子，即每候选 768 条轨迹。anchor 用单侧 80%
Wilson 下界做保守规划，而不是直接拿观察比例外推：

- B 的 144 个 anchor 中至少 6 个 blur/fusion bad 时，下界约为 `2.98%`，乘 768 后的
  保守期望约为 `22.9`，高于最终事件门 15；
- C 的 144 个 anchor 中至少 11 个 clear-bad 时，下界约为 `5.98%`，乘 768 后的
  保守期望约为 `45.9`，高于最终事件门 30。

这不是“保证一定获得”相应数量；它只是一个预先冻结的功效/成本停止规则。若两个候选
的类别集合完全不重叠，确认池最多 1,536 条完整轨迹；若重叠则更少。另用 seeds
`1100..1119` 做每类 20 条、无视觉标签的 class-specific 风险阈值校准；正式确认 seeds
为 `1200..1327`。

在把 endpoint 暂按独立 Bernoulli 处理的规划模型下，恰好落在上述 Wilson 边界时，正式
池达到 B≥15 与 C≥30 事件门的概率约为 96.9% 和 99.6%。这仍不是 AUC 检验的正式 power
analysis：确定性 seed、类别内相关和 reviewer 误差都可能使这个二项模型过于乐观。
旧第三池最终 6/1,800 与 blur 4/1,800 因标签可靠性失败，不得再用于发生率或功效估计。

## B/C 的隔离与最终事件门

- B 仍是九个固定 checkpoint 上的 `decoded_local_blur_severity__mean`，bad-high，主
  endpoint 仍是 blur/soft-fusion clear-bad 对 clean-good；
- C 仍是 sampling step 100–149 内 latent channel 3 的 alpha-compensated gradient-energy
  最大正跃升，bad-low，主 endpoint 仍是所有 clear-bad 对 clean-good；
- 两个单列 score product 必须物理分开；标签锁定后，B 若少于 15 个 blur/fusion bad，
  或少于 60 个 clean-good，或少于 3 个同时含 event 与 clean-good 的可比较类别，就不给
  evaluator 打开 B 文件并令其 family p 值为 1；C 对 30 个 clear-bad 使用同样的另外两门；
- 可打开的候选仍用 class-matched tie-aware AUC、固定方向置换检验和两候选 Holm 校正；
- 每次置换共同打乱 128 个完整 global-seed 标签块；B/C 各自在自己的 6 类范围计算，两个
  候选使用同一 seed permutation。旧协议的“三类别块”文字不再适用；
- 只有 B 通过原来的 AUC、Holm 和 operating-point 门，才允许做 blur/fusion 专项干预。

## Inception/DINO/FID 的边界

它们可以在人工标签完全锁定后作为 endpoint 外部旁证，例如描述某批 bad 是否也远离正常
样本表征中心。但它们不能参与类别排名、anchor GO、B/C 检验、在线预警或干预。单图也
没有可解释的 FID；FID 只适合比较足够大的两个样本分布。

## 结论范围

若通过，结论只能写成“在预先由独立 endpoint screen 选出的高风险 ImageNet 类别上，
冻结的 B 或 C 能预终点区分相应失败”。它不能外推成随机 ImageNet 类别、所有伪影或
所有扩散模型上的普适检测器。类别富集提高功效，同时必然缩窄 claim scope。

## 当前执行状态

这个锁冻结科学设计、selector 源码和可见教学 anchors，**尚不可真实开跑**。pair-keyed
singleton endpoint sampler 及四 GPU runner 已经实现、冻结并通过 source-lock、自测、
synthetic smoke 和 1,008-pair dry-run；真实输出目录仍不存在，采样数为 0。其 sampling
protocol / source manifest identity 分别是
`6e0a222ff313262f641b1a2b6d70cdfbc73fd71da844396a5d4db3974ceb58d3` /
`36cc1dfa90aff69737338f8970703c726a6794cc8a7677bbe4d1494ac8e40da4`。

真实采样前仍必须补齐并冻结：两专家 anchor ratification、隐藏
qualification/reserve forms 与 evaluator、三 reviewer + 双 adjudicator 全流程、绑定
anchor plan 的 dynamic full-trace runner、物理分开的 B/C score extractor，以及先只读
标签门、过门后才读取单个候选分数的两阶段 evaluator。当前缺少这些工件和真实合格评审
输入时，“不打开分数文件”仍只在部分采样组件中得到程序保证，不能把整条科学流水线称为
ready。

冻结工件：

- protocol identity: `04e933793992e2a7ce62aa4ac66836412f3c4f221cce731f2e072da97e892dd7`
- manifest identity: `0778e0ad2732256a1377d61ba7f04c6ad4f1fdca3a7fd9dec00d0b89e0247e36`
- selector: `experiments/select_dit_event_rich_classes.py`
- lock: `experiments/locks/dit_event_rich_confirmation_protocol_lock_v3`
- endpoint sampler: `experiments/sample_dit_imagenet256_endpoint_pairs.py`
- four-GPU runner: `experiments/run_dit_event_rich_endpoint_screen.py`
- endpoint sampling source lock:
  `experiments/locks/dit_event_rich_endpoint_sampling_source_lock_v1`
- superseded audit trail: `experiments/locks/dit_event_rich_confirmation_protocol_lock_v1`、`v2`
