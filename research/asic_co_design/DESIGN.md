# 从 KV Version Hierarchy 到专用 ASIC 数据流

更新：2026-11-11。定位：算法与 ASIC 协同设计；本文件是一份可证伪的方案及 CPU 初步实现，不是完成的硬件评测报告。

## 1. 主张与研究边界

建议保留论文问题 **When to Share, When to Prune**，将架构主线收敛为：

> 四轮 KV 的层级冗余把“许多独立版本的全局排序”，约束为“许多小树的合法切分”。能否直接在 Attention 统计流与树描述符之间求预算切分，避免反复生成、写回和排序全部版本分数？

潜在贡献是 **hierarchy-constrained budget cuts 与数据流共同设计**；不是移位器、阈值或优先队列本身。版本树独立于后续 Query，但树的 Attention importance 不独立于 Query。必须分别利用这两种时间尺度。

先解决版本管理器，再决定是否接入 predictive sparse attention。后者改变模型实际 Attention，不能用 Lazy Update 的选择一致性证明代替精度验证。GPU 上慢本身也不是 ASIC 贡献；必须证明专用控制、存储和数据路径实际减少硬件成本。

冻结目录 `experiments/streaming_hierarchy_band_guard60_*` 保持不变。此目录是独立实验分支。60% 压缩对应当前完整四轮 KV 的 40% payload；75% 是后续目标。元数据不占 payload 预算，但进入真实 SRAM/HBM、面积与能耗核算。

## 2. 已有证据，和不能从证据推出的结论

| 证据 | 实际支持 | 尚不能支持 |
|---|---|---|
| 原联合原型 dev64：GSM 47 vs FTS 46；MATH 42 vs 42；MBPP 30 vs 28 | 更高压缩率下值得继续验证；FTS 保留50%，候选保留40% | 全量显著优于FTS、或ASIC加速 |
| Experiment 8 合成轨迹128步，gate-first 当前分数刷新25.39% | 能重排计算并保持合成场景的联合选择一致 | 省75% Attention、或节能75% |
| 同一合成实验加入下界键计算后约92.84%的全量存活候选算分工作，另有大量堆访问 | 只看refresh ratio会误判收益 | heap一定适合硬件 |
| 真实cross-loop profiling：32条GSM、各32个decode query，采样1-based层6/12/18/24、16 heads | 后期loop的访问集合具有预测性 | 长生成、代码/数学、压缩缓存上仍成立 |

重新从原CSV统计，每条Prompt内平均采样Layer/Head，再对32条Prompt取中位数：

| 目标Loop | 上一轮Top10%→目标Top10%召回 | 上一轮Top10%覆盖历史Attention mass | 上一轮Top20%覆盖历史Attention mass |
|---|---:|---:|---:|
| 2 | 72.67% | 68.78% | 79.16% |
| 3 | 81.90% | 93.14% | 96.51% |
| 4 | 88.49% | 93.53% | 96.80% |

此处 **history mass 分母排除self**；旧报告的整行mass recall分母包含self，因此Top10%对应66.24%、89.55%、89.89%，两者不矛盾。查询的self必须单独加入执行集合。

尾部也重要：Top20%在Loop 2的head/layer/prompt单元平均history-mass recall的P10只有34.90%，Loop 3/4约93.04%/93.57%。这不是单Query P10，也不是未观测Token缺失质量的在线保证。结果见 `results/predictability_summary.csv` 和SHA来源记录。

## 3. V-Rex值得借鉴什么，哪里不能照搬

已阅读用户提供的本地14页V-Rex PDF。其§IV–V先从流式视频冗余和检索瓶颈出发，改变检索算法，再让专用检索数据流服务下一阶段KV访问；§VI把周期仿真、综合时序和存储系统模型接起来。尤其值得借鉴“中间表示是否容易消费”和“检索开销是否真的被隐藏”，而不只是列出硬件单元。

我们的对象是同一Token跨Loop的原生版本和不可逆树切分，不是视频Token聚类。不能复制Hamming clustering或WiCSum，把名称换成版本选择；预算约束是物理KV容量，不是累计Attention质量阈值。存储预算、预测访问预算也必须分开。

来源：[V-Rex，HPCA 2026](https://doi.org/10.1109/HPCA68181.2026.11408603)。本地PDF的SHA见来源清单；不上传论文全文。

## 4. 建议主候选：直接求树切分，避免生成全部分数

### 4.1 哪些量不动，哪些是独立消融

不动：4轮、原生BF16 KV、K/V独立连续树、Rightward Routing、当前Token使用当轮新KV、版本不可恢复、近期20%禁止Root pruning、当前上下文全模型payload预算。

先保留原六对relative L2和树拓扑，只改变score表示/selector。原构树发生在Token完成四轮时，不是每个decode step重算所有历史高维距离；不能把一次性成本夸大为每步成本。

**注意现有代码事实：** 当前60%候选的树在Layer/Head/Token/KV维度独立，但children使用loop/head平均累计mass，root使用loop/head累计mass最大值；root删除在深层相邻layer对上跨head一致。不能写成“已验证的head独立importance/局部root”。保持此精度基线，head独立统计作为单独后续消融。

### 4.2 定点分数与逆阈值

将每个固定树节点的clipped cost量化为 `q_j = 2^(-s_j)`，零值独立编码。coarse-to-fine次序下 `s_0 <= s_1 <= s_2`。第一组只量化原树cost，不改树拓扑。

维护固定公共标度的累计Attention整数 `M`；新的child utility定义为：

```text
U_j = M >> s_j
保留条件 U_j >= T  等价于  M >= (T << s_j)
```

因此，选择阶段可直接广播阈值到树描述符，返回合法保留深度0/1/2/3，无需写出每个child的浮点分数。K/V使用相同的公共Attention标度，不能分别归一化以人为消除残差差异。

在精确实数同一N下，除以N是公共缩放；**浮点实现存在rounding/tie差异**，Experiment 8已有反例。这里定义新整数算法，只要求与同整数定义oracle一致；不能说与旧FP32逐值一致。root–version exchange若接入此域，所有root/child/shadow也必须采用一致缩放。

可从Q0.16的逐步平均Attention增量、48bit无饱和累计量开始，随后消融精度。饱和会改变优先级；不得静默溢出。若改变scale/renormalize，所有历史证书必须失效或保守重算。这里只给设计起点，不宣称位宽已足够准确。

### 4.3 预算切分与层级合法性

一次决策冻结物理可用版本、Attention统计和Root集合。扣除存活Root成本后，得到可保留child slot数B。每棵树仅考虑当前物理可用的coarse-to-fine前缀，不能复活已释放child。

1. 流式读树描述符，比较逆阈值，累加可保留child数。
2. 调整全局阈值，使严格高于边界的数量不超过B，包含边界的数量至少B。
3. 边界同分使用固定ID顺序，树内父节点优先；得到恰好B个slot。
4. 提交树切分和native anchor映射，释放物理slot。

这与独立score排序在**同整数分数**下等价，因为每棵树的child分数单调，稳定tie顺序不破坏依赖。CPU实现 `selector.py` 已用1,924个合成预算用例对照完整排序；另有100组树/渐进路由用例。尚无真实轨迹或模型精度结论。

当前CPU用位宽次二分扫描，是功能基准，**不是最终高效实现**。新合成计数有7,345次当前mass读取，但约143万次证书标量读取，不能只报前者。下一步对比：前一步阈值窄区间搜索；不相交tile的范围证书跳过；只对边界tile细化。若分布突变导致区间失效，必须回退完整计数并计入P99。

### 4.4 Lazy从“缓存候选score”变成“缓存切分有效范围”

对一个树切分，其有效范围由最差已保留child和最好未保留child给出。对质量完全相同的固定阈值，只要累计mass仍在该区间内，切分无需改变。它比维护3个随时间更新的浮点priority更贴近最终预算动作。

保存mass下/上界。仅当某阈值落入不确定范围时读取精确mass；否则直接贡献计数。下界可由累计量单调性获得；上界可由已知最大增量×经过步数得到，但会快速变松。更紧界需要读统计或维护tile增量，成本要算进去，不能使用尚未读取的当前mass伪造免费证书。

全局阈值变化也会令切分失效；新Token、pruning、预算变化和slot释放必须更新generation/version tag。固定Snapshot内验证的证书不能跨错误世代复用。

**待验证主假设：** 真实树成本差异是否使多数树远离预算边界，从而证书持续有效，减少mass访问与选择控制，而非把score计算变成更多metadata扫描。

## 5. 第二候选：在原生KV产生时收集廉价树描述符

仅在选择器质量验证后，独立替换高维L2：从每个原生KV流固定采样m个坐标（候选m=16/32/64），所有四轮使用同一确定坐标集和共同定点尺度；计算六对sampled absolute differences之和及较早版本绝对值和。用阈值移位比较得到相对L1的dyadic编码，构建同样的连续complete-linkage树。

`scoring.py`是此候选的CPU功能模型；它不是L2的无偏估计，也不是输出误差界。成本在建树前量化会产生新ties并改变拓扑；必须与“只量化原树cost”的候选分开。零范数、极小非零差异、离群坐标和尺度溢出都有明确测试。

数据流在KV写出时旁路采样：subtract/abs/add、leading-bit/shift compare、小型3轮相邻merge比较器。无需维度乘法、平方根和除法。旧轮采样向量必须保留到第4轮；prefill并行Token会扩大临时scratch，不能假设每次只存一个Token。采样坐标预先固定，不能按测试集挑选最优坐标。

固定m只降低每个新树的构建工作，不直接降低历史Attention访问。若拓扑质量不够，则退回原L2构树，保留cheap selection；无需一起推翻Version Tree。另设全维integer L1以区分距离指标与采样误差。

## 6. Root pruning如何接入，而不偷换方法

第一步验证version-only选择；第二步在同一物理snapshot中复现原“一次provisional分配→root exchange→最终重分配”。provisional删除不落物理存储，最终commit才释放，否则会丢失重新分配本应保留的版本。

继续使用原max-root / mean-child定义、深层相邻对、近期保护和pruning cap。计算根代价包含当前暂保留child价值，只对已通过廉价峰值/近期gate的band读取与求和。不要把“令所有Root竞争一个heap”写成已实现机制。

整数域可比较 `band_loss < shadow * band_slots`，避免除法；slot数小且有界，但仍需小整数乘法或shift-add，不能声称全系统零乘法器。root cumulative max仍依赖各loop/head历史mass，维护成本可能大于版本算分，必须单列。

sharing和pruning动作、mask、budget、重新分配都要对照同整数完整oracle。带保护root的最低可行payload高于预算时应报不可行，不应悄悄突破预算。每次实际Payload在commit后审计；暂存/峰值另外报告。

## 7. Predictive Sparse Attention：可选扩展，不是既定贡献

[FlashLoop §3.2/§4.2](https://arxiv.org/html/2609.29812v1) 已根据前一loop的重要列预测后续列，并处理部分质量和输出更新。因此不能以“跨loop预测”单独主张新颖性，也不能保证无人做过。新机会在预测的逻辑Token集合如何与已有版本树的物理KV映射共同执行。

建议最小实验：Loop 1/2完整访问当前存活上下文，Loop 3/4分别用**同一Query在上一Loop**的Attention预测；不同Query的预测另设baseline。预测集合与存活root取交集，强制加入self和配置的近期访问窗口；访问窗口与不可prune窗口是不同参数，需报告实际并集大小。

先固定20%/30%历史候选做可复现诊断，逐层/head报告mass与漏失，然后才讨论自适应mass阈值。Loop 3一旦稀疏，Loop 4不能偷用dense参考的Loop 3统计。区分teacher-forced离线预测与真实causal rollout。

候选数据流：预测逻辑token → 树的rightward映射 → K/V独立物理slot请求 → residency/tag检查 → 读取原生KV → sparse QK/softmax/AV。相同原生版本可能在多loop重复访问；是否能跨loop留在片上、减少HBM，必须依据真实全模型loop执行顺序测量。中间隔着其他层，不可假设相邻loop访问马上相邻发生。Q改变时，即使复用同一K/V也不能直接复用attention结果。

先明确一种近似语义：对预测集合计算并重新归一化softmax。它会改变输出和累计importance，需单独测质量。与FlashLoop的质量校正/残差输出方案对照，不能默认等价。使用周期完整访问或抽样探测诊断预测漂移，但抽样不能保证发现所有漏失；除非给出有效界，不能声称精确fallback保证。

永久pruning和本次skip语义不同：预测未访问Token仍可物理存在，后续完整访问可重新被看见；已prune的root不可复活。若稀疏统计使未访问Token越来越低分，可能自强化误删，初期不同时启用预测访问和新pruning规则。

必须测量“KV存储保留率、逻辑访问率、实际物理读字节、总Attention计算量”四个量，不能用25% score刷新率代表75% Attention节省。

## 8. 微架构草案与费用

优先考虑共享描述符SRAM、阈值状态和commit接口的一体化管理流水线；新树构建可作为低吞吐旁路，不要求独立大模块。

```text
native KV写出 ─→ 固定坐标采样/树描述符 ───────────┐
Attention流 ─→ 定点统计累加 ─→ mass SRAM          │
                                                ↓
epoch+预算 ─→ tile证书检查 ─→ 逆阈值比较/前缀计数
                         ↖ 边界细化/全局count归约 ↵
       root gate/exchange ─→ 第二次合法切分 ─→ 原子commit
                                          ├→ slot free-list/map
                                          └→ 可选预测访问/物理KV请求
```

| 环节 | 单元与可流水化部分 | 可能主导成本的不规则部分 |
|---|---|---|
| 新树 | m路abs/add，少量移位、max、相邻merge比较器 | prefill scratch、定点尺度转换、收齐4轮的等待 |
| 统计 | softmax流旁路加法、定点量化、地址生成 | 同Token冲突、跨head累积max需历史mass读取 |
| 切分 | 描述符decode、受限移位、宽比较器、3bit前缀计数 | 阈值反馈、多遍扫描、证书过期、bank冲突 |
| 全局预算 | 分组count adder tree、标量反馈控制 | 跨PE归约与全局同步，不能随意按layer独立预算 |
| pruning | recent/peak gate、band求和、小整数乘法比较 | 变长slot释放，transaction rollback/commit |
| KV管理 | 原生slot地址映射、bitmap/free-list | 碎片、部分写、回收与Attention读的hazard |
| 预测访问 | 索引/bitmap、map查找、request合并 | gather、cache miss、跨loop复用距离 |

并行度从4/8/16个tile lane扫描，单lane输出局部count再归约。分组并行不是分预算。每一级register边界、SRAM端口数和阈值广播扇出要进入时序；预期关键路径可能是宽移位/比较+归约反馈或SRAM冲突，需综合/仿真后确认。

### 存储不能忽略

示例假设L=24,H=16,D=128,N=4096；不含batch，全部以脚本公式推导：

- Full原生BF16 payload为3 GiB；40%容量约1.2 GiB。
- 每树32bit描述符（3×4bit shift、3bit zero、6bit anchor、2bit cut、1bit live、8bit epoch）：全部K/V约12 MiB，**不是一个小寄存器表**。epoch wrap必须flush/revalidate。
- 共享48bit mass约0.5625 MiB；保留原root定义的4-loop/head累计48bit mass约36 MiB。可只对允许prune层维护，但这会要求清晰说明统计范围，不能随意丢弃历史。
- root peak取max可流式约简为每(layer,token)一个标量，不必存所有单头peak；但累计mass的max不能用同样方式删除所有单头历史量。
- 一个逻辑Token预测bitmap约0.1875 MiB；dense逻辑slot的32bit指针表上限约48 MiB。更紧凑slab映射需同时建模碎片和重定位，不能只报pointer减小。
- 描述符16遍完整扫描就是192 MiB/决策，尚未包括其他元数据。若多数描述符在HBM，这可能使候选失败。必须用tile驻留/边界过滤实测减少流量，或保留更简单控制算法。

因此探索256KiB/1MiB/4MiB管理SRAM时，必须以tile工作集运行，并统计溢出访问，不能假装全表片上。`resource_model.py`只计算字节，不给未经验证的面积或频率。

## 9. 验证顺序与停止条件

| 阶段 | 对照/实验 | 必须回答 |
|---|---|---|
| A 已有CPU初验 | 新integer selector vs integer排序oracle；独立sampled树 | budget/tie/层级/不可恢复正确？已有合成通过，真实trace未完成 |
| B 冻结真实trace | FP32树分数→只量化cost→只量化mass→整数selector；真实g/uniform g | 选择分歧、score误差、代价差异是否减少metadata总访问？ |
| C 低成本构树 | 原L2、全维integer L1、m=16/32/64 sampled L1 | 拓扑/边界质量与评分开销；不得同时改pruning |
| D joint replay | 完整root exchange vs新selector | shadow、root mask、最后mapping/payload逐步一致（同整数定义）？ |
| E 生成精度 | 冻结GSM/MATH/MBPP训练32条起步，必要时预定扩大64 | 完整回答和逐题差异，不能把dense recall当生成正确率 |
| F Sparse独立消融 | dense、仅新管理器、仅预测访问、组合；Loop2是否dense | 跨任务/长生成质量、统计反馈、实际访问字节 |
| G RTL | 先实现逆阈值/tree-cut PE和count controller，与功能模型逐cycle接口对照 | stall/backpressure、overflow、tie、epoch、长尾bank冲突 |
| H 综合/功耗/系统 | 同工艺同精度基线、SRAM宏或明确替代模型、活动因子 | 面积/时序/动态静态功耗、每决策能量、E2E收益 |

B阶段优先重放已有真实统计；若缺少完整轨迹，CPU合成测试只能验正确性。新GPU工作必须先核对资源状态；本次未提交GPU或大规模benchmark。

硬件baseline：FP32全量评分+选择；同定点score全量扫描；generic heap lazy；同SRAM/精度的streaming threshold；专用tree-cut；再分别开/关证书和预测访问。否则无法区分低精度、通用selector改进与hierarchy贡献。Uniform cost/attention仅机制诊断，不当作原模型准确率。

精度同时对照Full KV、FTS及现有联合原型；正式宣称前使用冻结全量协议与逐题统计。失败时定位cost量化、mass量化、阈值/tie或sparse反馈，不能在96条结果上反复调公式后宣称已验证。

## 10. 真正的ASIC评估应交付什么

周期模型记录FIFO深度、端口数、bank映射、读写冲突、依赖、重试、partial-line访问、commit等待和P50/P95/P99。只用操作数除以PE数是下界，不是cycle simulation。版本决策速率用decision/s，模型吞吐用token/s，分开报告。

RTL先实现关键路径，之后综合完整控制与SRAM接口。固定工艺库、PVT、时钟约束；报告组合/时序单元与SRAM面积，不把SRAM当免费black box。功耗用代表性真实trace生成SAIF/VCD，计算logic+SRAM+外存总能量；没有活动数据不能只用gate count推低功耗。

能耗账本为各类访问次数×对应访问能耗，加运算、时钟、静态功耗；系数需有来源。HBM以实际事务字节计，不以逻辑KV元素数代替。area efficiency与energy efficiency在相同正确率/预算/工作负载下比较。

E2E模型保留全部四轮MLP/QKV投影、attention、KV访存、管理器、通信、预测和fallback。用事件依赖图判断重叠；prefetch只是提前请求，不自动代表完全隐藏时延。不同batch/上下文长度、带宽、片上容量、loop数量均要扫点。

**接受条件：** 同等质量与payload下，加入证书维护、统计、mapping后总周期/能耗仍降低，且优势不被一个同预算通用threshold基线消除。否则主张应降为算法优化或放弃该硬件方案。现阶段面积、功耗、频率和E2E加速均为未验证，不填虚构结果。

## 11. 当前优先级

1. 先验证“树切分+逆阈值”在原cost上能否减少真实总metadata工作，而不是马上改L2。
2. 再验证streaming sampled-L1是否保住树的质量。
3. 将预测访问作为独立可插拔扩展，优先后两轮；只有与物理版本布局产生额外可测收益时，才作为协同设计主贡献之一。

不能保证最后一定成功或一定具有足够新颖性；但每个方向都有对应的真实观察、硬件瓶颈、独立实验与失败判据。
