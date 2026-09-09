# 官方SiT-XL：OU方向是否比raw PFR更好

当前继续PFR，fixed point与论文暂停。已核对：官方XL原始PFR有独立
5K正结果，但尚无OU额外方向收益的官方XL直接对照。小SiT v/x的
已有OU改善不能自动转移到联合训练Base的官方XL。

本轮固定两组1K，继承官方PFR seed202609428、B4、100步uniform
Euler、FP64状态/FP32模型/noTF32、IG1.35、h1/32、noise-time>.5。
raw全程保持原时间差分；ou只在noise-time>.75时，将原弱时间差分
投影到同状态标准化OU坐标下的Full degree1 defect，再恢复原逐样本
RMS；.5<t<=.75仍用raw，其他时刻无修订。无新强度/窗口/方向搜索。

直接复用小SiT的split_raw_revision_against_ou_degree1和
rms_match_per_sample；noise time到data time的变换及速度符号，使用
已有raev2_pfr_retiming里的通用noise-time包装。这里使用它是坐标
约定相同，不调用RAE模型。OU查询的state/t均按当前模型FP32输入。

每图两组均125 Full+50 prefix，含25额外Full查询；raw丢弃OU查询输出。
这是同查询数量机制对照，可能有实测时间差；不是对最快raw版本的
效率优势。预计两组约.7 GPU小时采样，加载/预检/评价另计。

每驱动先native8和raw8逐像素复现，再各自smoke8与实际future-prefix
核验。正式前8复现smoke；raw正式1000张必须复现旧raw整个像素bank。
记录源码快照、权重、VAE、输入、像素hash、调用数、原官方评价特征。
完成后独立FP64重算FID，并核对上述来源与预算。

这是固定1K方法比较；如果OU有优势，仍须独立5K与相应实测成本对照
才能声称可靠提升。旧raw独立5K不为OU背书；也不能据XL结果解释RAE
失效的单一原因。不以本轮结果修改方向、h、scale或窗口。

脚本sample_official_sit_pfr_ou.py和run_official_sit_pfr_ou.py。

## 执行状态

驱动30991（raw，GPU1）、86395（ou，GPU2）均exit0，正式1K和评价
完成。独立核验5359 exit0：所有输入/源码/像素/预算检查通过，raw全部
1000张逐像素复现旧raw，FP64 FID最大重算差2.7e-5。

CPU坐标/符号预检58724 exit0：同一N(mu,I)数据端的精确velocity，
在noise-time1/.9/.8和未来1/32处，标准化状态输运精确匹配闭式，
degree1 defect绝对值最大1.78e-15。该高斯例子本应只含被消去的
OU degree1相对score模式，验证转换接口；不是神经模型或质量证据。
结果official_sit_ou_gaussian_interface.json保存在便携结果目录。
逐样本投影和RMS恢复沿用原实现，包括零投影时返回零的退化约定。

## 完成结果

| 方法 | FID1K | 均值项 | 协方差项 | IS | 采样秒 |
|---|---:|---:|---:|---:|---:|
| raw（相同额外查询） | 40.565427 | .728290 | 39.837163 | 52.141123 | 1234.078014 |
| OU方向 | 40.554934 | .621245 | 39.933716 | 53.192445 | 1231.864004 |

FID仅降低.010492（约.0259%）；均值项降低.107045而协方差项提高
.096553，几乎抵消。当前没有可靠的额外质量优势证据；这不是统计
等效性检验，也不证明总体OU效应为零。不能把旧raw5K改善归入OU。

两组各125 Full+50 prefix/image，查询数相同，实测时间相近。
raw是为配对而执行并丢弃额外OU查询；最快既有raw只需100 Full+
50 prefix，故本轮更不能声称OU具有效率优势。采样合计2465.942018秒，
.684984 GPU小时；加载、预检、评价和CPU核验另计。

研究决策：不因本轮微小差距追加OU5K或调方向/窗口。原始PFR的
官方XL独立5K正结果仍保留；小SiT上的OU额外收益不能直接推广到
官方XL。不能仅凭此比较把差异归因为联合训练、容量或数据集之一。
本轮填补方法对照缺口，但没有产生可宣称的新方法突破。

原始数据目录official_sit_pfr_ou_20260908；便携审计结果
experiments/results/terminal_defect_20260908/official_sit_pfr_ou.csv。
两个驱动已退出，本实验没有后续活动进程。PFR路线继续，fixed point
单独暂停，论文暂停。
