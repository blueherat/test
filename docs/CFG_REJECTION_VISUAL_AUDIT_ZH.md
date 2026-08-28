# CFG-Rejection / EDM2 / ADM64 视觉 bad-case 审计规约

## 一、不能再用“类别大致可辨认”代替质量判断

图像能被识别为 golden retriever、panda 或 strawberry，只说明语义类别大致
匹配。它不能排除以下 bad case：

- 眼睛、口鼻、牙齿、肢体、果实轮廓的错位或形变；
- 多个主体、部件或物体相互粘连，或者出现缺失、重复和错误连接；
- 主体与背景、植被、木头或地面融合；
- 局部模糊、液化、过锐、塑料质感和重复纹理；
- 小物体、文字样伪影、边界断裂或不符合生长、遮挡、接触和支撑关系的结构。

因此，之后不得根据缩略图、类别可辨认性或第一眼观感写“没有明显 bad
case”。`semantic_match`、结构可评估性和伪影严重度必须分开记录。

## 二、每张图必须完成的七区检查

正式工具要求每一项选择 `clear / defect / unassessable / not_applicable`，七项
未全部完成时不能锁定标签：

1. 主体数量、整体姿态、场景布局和完整外轮廓；
2. 脸/头，或目标物体、植物的主几何；
3. 肢体和部件的数量、起点、终点、关节、解剖与对称性；
4. 主体之间及部件之间的每处接触、遮挡、共享边界和连接；
5. 主体与背景的边界、支撑、深度顺序和接触关系；
6. 局部清晰度、液化、过锐、重复纹理、颜色/光照斑块和文字样结构；
7. 背景中的次要物体、图像边缘、裁切和小尺度结构。

选择 `not_applicable` 时必须在 notes 解释为何该检查对这张图确实不适用，不能把
看不清或不愿检查的区域记成 N/A。

`unassessable` 不是 `clear`。只要关键结构因主体过小、遮挡或像素支撑不足而
看不清，就必须把 `structural_assessability` 记为 `partial` 或
`insufficient` 并写明原因；这种图片不能取得 severity 0。

### 强制视图

- 大于 64 像素的图：必须看原生 `100%` 和平滑 `2x` 分区视图；
- 最大边不超过 64 像素的图（当前 ADM64）：默认以 nearest-neighbor `8x`
  打开，随后还必须看 smooth `8x`；两种视图缺一不可；
- nearest `8x` 用于检查原生像素中的拓扑和边缘，smooth `8x` 用于检查实际
  感知观感。只在 smooth 插值后出现的问题不能直接归因给模型；反过来，像素
  支撑不足也不能被写成“无问题”。

工具会记录视图事件及时间。记录只是完整性证据，不能代替人工逐区检查。

## 三、严重度、硬结构缺陷和锁定规则

- 0：充分可评估，且没有任何可指认结构或成像问题；
- 1：单处轻微成像或纹理问题，整体结构仍连贯；
- 2：存在清楚可定位的错位、粘连、形变、拓扑、边界、模糊或伪影；
- 3：主体主结构失败，或存在多处独立、明确的缺陷。

以下 flags 是硬结构缺陷，工具会自动把最低严重度升到 2：

- `face_or_head_geometry`、`limb_or_body_geometry`、
  `object_or_plant_geometry`；
- `object_boundary`、`object_fusion`、`multi_subject_fusion`、
  `subject_background_entanglement`；
- `subject_missing_or_too_small`、`human_anatomy`、
  `missing_or_extra_parts`；
- `topology_attachment_or_contact`、`perspective_or_support`。

severity 大于 0 时必须写出缺陷的位置和具体表现，不能只勾一个宽泛 flag。
severity 0 必须同时满足：结构可评估性为 `sufficient`、七区检查没有
`defect/unassessable`、没有 artifact flag，并选择 `none_observed` 成因判断。

任何 severity 0/1 标签都必须：

1. 先锁定第一次判断；
2. 离开该图、检查至少另一张图后再返回；
3. 重新完成该分辨率要求的两种视图；
4. 再次检查并锁定第二次判断。

否则导出只能是 `draft_incomplete`，不能进入正式统计。最终标签记录
`initial_locked_at`、`secondary_reviewed_at`、`label_locked_at` 和完整
`view_events`。锁定后若重开，必须重新完成整套流程。

## 四、ADM64：模型伪影与分辨率上限必须分开

`artifact_origin_judgment` 是可审计的人类判断，不是模型成因真值：

| 判断 | 使用条件 |
|---|---|
| `model_likely` | 不可能的轮廓、连接、部件数量、主体融合或背景桥接在 nearest 原生像素中已经存在；或某局部相对同尺度邻域发生选择性液化/崩坏。 |
| `resolution_limited` | 全图一致的低带宽、锯齿或微细节缺失，或者关键部位像素支撑不足。应同时标 `partial/insufficient`，不能降成 severity 0。 |
| `rendering_only` | 问题只在 smooth `8x` 插值后出现，nearest `8x` 的原生像素中没有相应结构。 |
| `natural_occlusion_or_crop` | 普通遮挡或构图裁切足以解释观察到的局部现象；必须写出理由。 |
| `uncertain` | 观察到问题或可疑区域，但证据不足以区分模型、分辨率和自然遮挡。 |
| `none_observed` | 仅可与充分可评估的 severity 0 同时使用。 |

小文字仅仅因为只有几个像素而不可读，属于分辨率限制；较大文字区域出现重复、
断裂或伪字笔画，才是模型伪影候选。均匀的低清晰度是分辨率本底；只有脸或肢体
相对同尺度区域异常熔化，是模型伪影候选。后续应把类别和主体占比匹配的真实
ImageNet 64x64 控制图混入盲包，用来估计分辨率本底，但真实图不能被当作逐张
“无缺陷”真值。

## 五、绝对缺陷不等于模型相对 bad

当前 `build_blind_bad_case_audit.py` 的 schema v2 同时导出：

- `defect_present = artifact_severity >= 1`；
- `clear_bad = artifact_severity >= 2`；
- `semantic_bad = semantic_match == no`；
- `overall_bad = clear_bad OR semantic_bad`；
- `possible_bad = NOT overall_bad AND (severity == 1 OR semantic uncertain OR
  assessability partial/insufficient)`；
- `clean` 只在最终锁定、severity 0、semantic yes、assessability sufficient
  同时成立时为真。

这些是 v2 的**绝对缺陷字段**。其中 `clear_bad` 这个旧名字容易误导；它只表示
“能明确定位绝对缺陷”，不表示这张图明显低于同一模型的通常采样水平，也不再
作为正式质量主终点。以后在分析表中把它解释为
`absolute_clear_defect = artifact_severity >= 2`。一张典型 ADM64 图完全可能有
`absolute_clear_defect=true`，却没有资格被判成模型相对 bad。

severity 1 必须叫“存在轻微缺陷”，不能在文字结论中偷换成“正常图”；反过来，
severity 2/3 也不能脱离同模型参考图直接偷换成“低于采样平均水平”。
`uncertain` 和 `unassessable` 不能塞进 clean 分母。直到下面的 v3 相对协议实现并
完成盲评之前，所有现有 smoke 的 `relative_bad` 一律记为 `not_evaluated`。

### v3 主终点：`model_relative_catastrophic_bad`

v3 保留全部 v2 绝对观察，并新增以下互不替代的字段：

- `absolute_defect_severity`、`absolute_defect_flags`、
  `absolute_failure_impact`、`structural_assessability`、
  `semantic_match`、`artifact_origin_judgment`、七区检查和位置说明；
- `reference_set_id`，以及冻结的 `model_hash`、`sampler_hash`、分辨率和
  `class_id` 匹配信息；
- 五条 `pairwise_comparisons`，每条保存匿名 anchor ID、
  `candidate_vs_anchor` 和 `comparison_assessability`；
- 派生的 `worse_count`、`clearly_worse_count`、`reviewer_relative_bad`、
  `adjudication_status` 和最终 `model_relative_catastrophic_bad`。

其中 `absolute_failure_impact` 固定取
`none / local_minor / material / primary_structure_failure`；它描述缺陷对主体的
影响范围，不描述候选图相对 anchor 的名次。`comparison_assessability` 固定取
`sufficient / partial / insufficient`，并与单图的结构可评估性分别记录。

每个候选图与五张**同权重、同 sampler 配置、同分辨率、同类别**的冻结 baseline
(P) 典型 anchor 比较。若是文本条件模型，应进一步同 prompt；不能用方法臂自身
的中位数重新归一化。比较采用
`-2 明显更差 / -1 稍差 / 0 相当 / +1 稍好 / +2 明显更好`。单名审核者只有同时
满足以下条件，才能给出 `reviewer_relative_bad=true`：

1. 相对比较充分可评估；
2. 五次中至少四次为候选图更差，其中至少三次为 `-2`；
3. 有可定位的 `material` 或 `primary_structure_failure`，而不是仅有同分辨率
   图普遍存在的粗糙；
4. 问题在规定的原生/nearest 视图中有依据，不能只是 `rendering_only`。

两名审核者均为阳性时形成临时阳性。所有临时阳性、所有跨主阈值分歧都交给
第三人盲仲裁；另随机抽至少 10% 双阴性检查共同漏判。最终主终点只取
`model_relative_catastrophic_bad`。绝对 severity、缺陷类型和 `possible_bad`
继续作为缺陷负担、机制分层及敏感性分析，不能再决定主标签。

## 六、参考库、双人独立复核和仲裁

anchor 必须来自与待评估 seed 完全不重叠、且不按 ASD、路径证据、触发状态或
尾部排名挑选的 baseline-P 参考库。先由独立校准审核者做类内盲成对排序，再从
估计质量分布中央 40%–60% 的稳定样本中冻结典型 anchor。每个候选图使用五张
anchor；左右位置和顺序分别按 reviewer salt 随机。绝对缺陷单图检查应先锁定，
之后才进入相对比较，避免 anchor 改写已经看到的局部缺陷。

若 sampler 的随机流只由 seed 而不是 `(class, seed)` 决定，正式参考库和评估集
应给每个类内重复分配唯一 seed，避免不同类复用完全相同的 innovation 路径。
只有专门研究跨类 seed 混杂的诊断块才允许复用；历史上已经复用的结果必须按
seed 聚类分析，不能把每张图当作独立样本。

- 快速 pilot：12 类，每类 32 张独立 baseline 参考图，另取每类 20 个新的
  baseline/method 配对 seed；只用于调协议和判断是否值得扩大；
- 正式 discovery：50 类，每类至少 40 张独立参考图。若 bootstrap 中任一选中
  anchor 有超过 20% 的概率落出中央 30%–70%，该类扩到 80 张后重新冻结；
- confirmation：50 类 × 50 个未查看的配对 seed，共 2,500 对
  baseline/method 输出；reference、discovery 和 confirmation seed 三者互斥；
- 参考库只用 baseline P 建立，并同时评价 baseline 与方法臂。不能分别按两臂的
  “平均水平”选 anchor，否则真实的整体改善会被相对化消掉。

- 两位审核者使用不同 salt、匿名 ID 和随机顺序，独立完成全部图片，锁定前不得
  讨论，也不得看到 seed、ASD、路径证据、尾部排名或采样方法臂；
- 正式评估前先在不含评估图的 60 张校准集中考核：20 张专家共识模型相对灾难
  失败、20 张典型输出、20 张边界样本。启用门槛是灾难失败召回至少 90%、典型
  输出误判不超过 10%，且关键 topology/anatomy 样本最多漏 1 张；
- 必须同时报告主终点正向一致率、阴性一致率和加权 severity 一致性，不能
  只报容易被类别比例抬高的总体一致率；
- 以下情况一律进入第三人仲裁：跨 severity 2 阈值、severity 相差至少 2、任一
  人给 hard flag、semantic/assessability/origin 不一致、任一人给 uncertain；
- 另外随机抽取至少 10% 的“双人均为 0/1”样本给第三人复查，用来估计共同漏判；
- 仲裁者先独立标注，再看匿名化的原有位置说明/文字理由。一个审核者指出的具体硬
  缺陷，只有仲裁者复查同一区域并明确说明为何属于分辨率、自然遮挡或裁切时才能
  清除；否则保留为 `possible_bad`；
- 若校准召回低于 90%、典型图误判高于 10%、主终点正向或阴性一致率低于 80%，
  或抽查双阴性中超过 5% 被第三人升为模型相对 bad，应修订规约后整批重标，
  不能只修补分歧样本。

这些门槛先用于 discovery 校准，数值在 confirmation 前冻结。

## 七、smoke 的纠正性复查

24 张 smoke 图像的首次“没有明显缺陷”判断是错的。原图复查后，每张至少
都能指认局部问题；17 张为 severity 2，5 张为 severity 3，2 张为 severity
1，即 22/24 达到 v2 的 `absolute_clear_defect` 阈值。例如：

- class 207 / seed 3：人脸和手严重变形，狗嘴与所衔物粘连；
- class 207 / seed 5：两只狗的身体、脸和肢体在遮挡处融合；
- class 388 / seed 6：眼睛和口鼻错位，躯干与木头/阴影融合；
- class 949 / seed 4 和 6：果实形状粘连，表面液化、重复孔洞与过锐纹理。

逐图记录在
`experiments/annotations/cfg_rejection_edm2_smoke_exploratory_v1.csv`。复查发生在
看过信号网格之后，所以只能作为 `exploratory_nonblind` 定性材料，不得估计正式
TPR/FPR 或选择阈值。尤其不能从 22/24 推断模型相对 bad 率；这些样本没有冻结的
同模型典型 anchor，故 `relative_bad=not_evaluated`。

### exploratory CSV 与 v2 工具词表不能直接合并

CSV 的 24 行主键唯一、notes 完整，但旧 CSV 使用 18 种临时 flag；其中 13 种
不属于 v2 工具词表，23/24 行至少含一个不兼容 flag。因此禁止直接 concat、按
字符串计数或把它当成 v2 标注。若只为探索性汇总，可按下表重编码；
`local_geometry` 必须依据 notes 人工判定，不能自动映射：

| exploratory v1 | v2 controlled vocabulary |
|---|---|
| `blur` | `blur_smear_or_liquefaction` |
| `body_geometry`, `limb_geometry` | `limb_or_body_geometry` |
| `color_artifact` | `color_or_lighting_artifact` |
| `face_geometry`, `mouth_geometry` | `face_or_head_geometry` |
| `object_geometry`, `plant_geometry`, `surface_geometry` | `object_or_plant_geometry` |
| `repetition`, `texture` | `texture_or_repetition` |
| `text_artifact` | `text_or_glyph_artifact` |
| `local_geometry` | 依据 notes 人工重标；禁止自动映射 |

已经与 v2 同名的 `human_anatomy`、`multi_subject_fusion`、`object_boundary`、
`object_fusion`、`subject_background_entanglement` 可原样保留。任何重编码结果仍须
标为 `exploratory_nonblind_recode`，不能冒充正式盲标。

## 八、discovery 与 confirmation 边界

- 分类器只提供类别一致性弱标签，不能判断肢体、粘连、局部模糊或背景侵入；
- 正式盲包必须混合低尾、高尾、随机样本以及将来的 baseline/method 臂，且隐藏
  来源；不能用分开的包泄漏方法，也不能只展示有利尾部；
- discovery 可用于改规约和选阈值；confirmation 只能按锁定方案查看一次；
- 当前 `build_blind_bad_case_audit.py` 是**锁协议前版本**，会在读取任何源图之前
  无条件拒绝路径/manifest 中含 `confirmation` 的数据和 seed >= 10000。该限制
  目前不可覆盖；这意味着现在不能生成真实 confirmation 审图包；
- 只有在 rubric、采样计划、主终点/阈值、双人规则和仲裁规则全部冻结并留下版本
  记录后，才能通过新的、可审计的 schema 版本加入 confirmation 访问。不得靠
  改目录名、复制文件或删除 guard 绕过。

## 九、ADM64 六图 smoke 的第二次纠正

对 `adm64_guided/smoke` 的 3 类 × 2 seeds，首次复看仍错误地把两张狗图概括为
“相对连贯”。按本规约同时查看 nearest 8x 与 smooth 8x 后，6 张均有可定位的
缺陷或关键结构不可充分评估：

- class 207 / seed 0：颈肩到躯干的轮廓连接不自然，前肢与胸部结构缺失；
- class 207 / seed 1：胸腹形成无清楚关节和前肢终点的毛团，身体结构被局部
  模糊掩盖；
- class 388 / seeds 0–1：身体、树枝、植被和背景共享错误边界，肢体结构缺失或
  被融合关系替代；
- class 949 / seed 0：多枚果实互融，叶片、果蒂和果实的连接关系混乱，并有
  重复/液化纹理；
- class 949 / seed 1：主体边界和表面纹理大面积涂抹，果蒂缺失，局部高光与籽粒
  呈重复孔洞状。

这些记录已暴露 class/seed，只能标为 `exploratory_nonblind`，不能代替两位独立
审核者的正式标签。它们的作用是校正审核灵敏度：类别可辨认、主体占画面大、
第一眼像照片，都不能覆盖局部结构缺陷。由于没有冻结的同类 baseline anchor，
这 6 张只能记为 `absolute_defect_present=true`、
`relative_bad=not_evaluated`，不得再写成“6/6 都是模型相对 bad”。

## 十、FKC 发布配置 smoke：高照片感仍不能放行

用 FKC 发布的 EDM2-XS `0.045` conditional/unconditional、CFG 1.4、64 steps、
`S_churn=40` 跑出的 8 张普通 CFG 图，原图 100% 的探索性严审为：

- seed 0 / bluetick、seed 2 / black-footed ferret、seed 4 / letter opener、
  seed 7 / limousine：severity 2；
- seed 1 / hammer、seed 3 / can opener、seed 5 / otter、seed 6 / space bar：
  severity 3。

典型漏判风险包括：bluetick 只有一只眼选择性塌陷，但其余毛发极有照片感；
otter 群像纹理逼真，但多个头、身体和肢体无法一一连接；limousine 的大轮廓
完整，但车牌/饰条是伪文字，侧镜悬浮、轮毂错位、挡风纹理液化。8/8 都不能
进入严格 `clean`，但这只是一项绝对缺陷结论。

固定额外 CPU resampling seed 后得到的一张 FKC class-164 图整体比相同起始
seed 的 CFG 图连贯，但下唇有孤立肉片，项圈/吊牌连接含混，胸肩有旋涡式重复
毛纹，探索性暂记 severity 1 / possible bad，而不是 clean。更重要的是官方 FKC
额外抽取一次未使用的类别随机数并做 63 次重采样，最终第 1 槽的祖先未知；它和
CFG 图不是严格配对反事实。单张观感改善不能写成 FKC 质量提升。

以上图均已看过方法和 seed，只能用于 reviewer calibration。正式 FKC 对比仍须
把 baseline/method 图混入同一匿名盲包，记录 ancestry，并用独立样本估计提升率。
在同配置 baseline 参考库及 v3 anchor 比较完成前，这 8 张 CFG 和 1 张 FKC 的
`relative_bad` 都是 `not_evaluated`；不能把“8/8 有绝对缺陷”写成“8/8 低于该
采样器平均水平”。
