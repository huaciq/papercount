# P1：Gaussian / Bi-Gaussian Ridge–Valley 公式与实现核查

本轮为独立1D Shadow实现，未连接正式计数。以下区分“作者公开全文可确认的公式”“根据公式推导的数学性质”与“未给出的数值细节及工程选择”。不宣称完整实验数值复现成功。

参考：Tiejian Chen, Yaonan Wang, Changyan Xiao, *An Apparatus and Method for Real-Time Stacked Sheets Counting With Line-Scan Cameras*, IEEE Transactions on Instrumentation and Measurement, 64(7), 1876–1884, 2015。DOI：[10.1109/TIM.2014.2366977](https://doi.org/10.1109/TIM.2014.2366977)。使用[作者公开全文](https://www.researchgate.net/publication/273296062_An_Apparatus_and_Method_for_Real-Time_Stacked_Sheets_Counting_With_Line-Scan_Cameras)第III节（接受稿页3～6）的公式与步骤。全文文本可读取；本地公开页面请求403，PDF下载入口也不可用，未取得PDF供像素级公式核对。公式分段和符号有全文及连续性推导的双重检查，未发布阈值仍不能据此补齐。

## 1. Gaussian 多尺度 ridge likelihood（III-A，式1～2）

\[
G(x,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{x^2}{2\sigma^2}\right),\qquad
G''(x,\sigma)=\left(\frac{x^2}{\sigma^4}-\frac1{\sigma^2}\right)G(x,\sigma).
\]

\[
h_G(x,\sigma)=-\sigma^{2\gamma}G''(x,\sigma),\quad \gamma=1,\qquad
F_G(y)=\max_{\sigma_{\min}\le\sigma\le\sigma_{\max}}(I*h_G)(y).
\]

负号使亮纸身中心响应为正。σ范围对应纸身最小/最大宽度的一半，不能把相邻纸中心间距或周期直接当纸身宽度。论文展示的σ=3～5、5个等间隔尺度仅为特定图像示例。

实现 `gaussian.py`、`ridge_valley.multiscale` 同时保存每个尺度的卷积响应、逐点最大值、argmax索引与最佳σ；相同σ重复项去除，不将跨尺度峰数量相加。

## 2. 真正的分段 Bi-Gaussian（III-B，式3～6）

令 \(\sigma_b=\rho\sigma\)、\(k=\sigma_b^2/\sigma^2=\rho^2\)。分段原核为：

\[
bg(x,\sigma,\sigma_b)=
\begin{cases}
G(x,\sigma)+\dfrac{e^{-1/2}}{\sqrt{2\pi}}\left(\dfrac{\sigma_b}{\sigma}-1\right)\dfrac1\sigma,& |x|\le\sigma,\\
kG(|x|+\sigma_b-\sigma,\sigma_b),&|x|>\sigma.
\end{cases}
\]

其二阶导数是：

\[
bg''(x)=
\begin{cases}
G''(x,\sigma),&|x|\le\sigma,\\
kG''(|x|+\sigma_b-\sigma,\sigma_b),&|x|>\sigma.
\end{cases}
\]

\[
h_B=-\sigma^{2\gamma}bg'',\quad \gamma=1,\qquad
F_B=\max_{\sigma}(I*h_B).
\]

代码直接实现分段函数、解析一阶及二阶导数，未用DoG代替。ρ=0.3仅作为统一初始值。

**独立数学核查**：记 \(\phi=e^{-1/2}/\sqrt{2\pi}\)，在x=σ两侧，函数值均为ρφ/σ，一阶导数均为−φ/σ²，二阶导数均为0；x=−σ由偶对称得到相应结果。因此连接是C²。ρ=1时原核、二阶导数和离散权重均退化为普通Gaussian。尺度变换满足γ=1时 \(h_B(ax,a\sigma)=h_B(x,\sigma)/a\)。测试数值微分原函数，而不是只比较同一导数函数的两个调用。

这里的归一化指σ^(2γ)导数尺度归一化。未额外对Bi-Gaussian进行单位面积或L1幅度归一化，否则会改变它与valley项的相对权重。

## 3. 独立 valley 与 Ridge–Valley（III-C，式7）

\[
h_V(x,\sigma_v)=+\sigma_v^{2\gamma}G''(x,\sigma_v),\quad
F_V=I*h_V,\quad \sigma_v=\tfrac12\operatorname{mean}(\text{gap width}),\quad
F_{RV}=F_B-\eta F_V.
\]

这是独立单尺度valley滤波，正号使暗缝产生正响应。不是F_B取负，也不是仅保留ridge负响应。F_V保留符号，组合前后均不额外裁剪；只有峰识别与区域划分使用响应正部分。η=0.8为统一初值，未按真实GT挑选。

`ridge_valley.py`分别输出F_B、F_V、F_RV和各尺度F_B。最佳尺度来自F_B，valley只用单独σ_v。

## 4. 峰识别与正区域验证（III-D）

峰识别按正响应滑窗处理：半窗口σ_min；中心为局部最大时记录并前进σ_min+1，否则前进1；RAW中心灰度低于参考区域均值时剔除。代码保留识别前灰度拒绝及理由。

正文先提出中心间距大于2σ_min的先验，但随后给出的扫描跳步为σ_min+1。实现采用明写的扫描过程，没有自行加入2σ_min硬抑制并称之为论文要求。

区域验证按零交叉将正响应分成连通区域，每区域最多保留一个候选；再按区域面积A_r和最大响应r_m门槛剔除弱区域。原文没有公布这两个数值门槛、选择同一区域哪个候选或面积离散化细节。

本轮工程设置为：在含已通过灰度门槛候选的正区域上，分别取面积/峰值中位数，阈值取其0.1倍。**面积或峰值任一不达标即拒绝**，同区域保留响应最高者，相同响应取较小Y；这些决策规则并非已确认的论文原参数。0、0.05、0.1、0.2及面积/峰值不同组合的合成敏感性结果全部保存，不依据真实图计数选优。

## 5. 尚不能准确确认的实现细节

| 项目 | 论文可确认 | 本轮选择与限制 |
|---|---|---|
| 纸身/间隙宽度获取 | σ与真实纸身/暗缝宽度关系 | RAW固定0.7px轻平滑，P10/P90中值亮度阈值，测连续亮/暗区；宽度估计本身未经人工纸身GT验证 |
| 最小/最大宽度 | 最小/最大纸身宽度的一半 | 合成metrology用独立生成几何宽度极值；自动测量用亮段宽度P10/P90减弱噪声离群，属于工程偏离，保留完整宽度列表与极值 |
| 数字滤波 | 连续卷积和σ²γ归一化 | 每像素区间积分解析导数，5σ尾截断，中心点扣除微小DC残差，不做L1归一化 |
| 亚像素窄负叶 | ρσ的背景尺度 | 用像素积分避免ρσ较小时点采样丢失负叶；不是论文已说明的采样规则 |
| 参考区域 | 使用参考区平均灰度；实验成像保留两端背景 | 真实图用未裁切中心列两端各5%；合成用信号两端各5%。中心I_p取RAW单像素，窗口及选择未在原文给足 |
| 小数σ窗口与平台峰 | 窗口2σ_min+1及扫描跳步 | 半窗口向上取整，等值局部最大依扫描顺序处理，跳步后剩余重复由区域验证处理 |
| 端部与卷积边界 | 第III节未完整定义 | reflect卷积仅作数值延拓，真实支撑不够标记complete_support=false，触发review；不把反射像素当实际观测 |
| 正区域面积 | 对A_r及r_m设门槛 | 线性零交叉与梯形积分；区域不完整标记truncated；阈值选择见上节 |
| 饱和、黑带、低对比 | 未给通用像素可观测性标注协议 | 低对比/无法测宽输出insufficient_evidence；多饱和或边界支持不足输出review_required；仍保留候选，不在黑带按周期补纸 |

这些不确定项已经进入配置、日志与报告。连续公式及方法结构有证据支持；**不应将工程阈值设定或本项目成绩写成论文实验的严格数值复现**。

## 6. 独立测试与输出

数学/行为测试涵盖C²连接、ρ=1退化、数值微分、像素积分与数值积分一致、常量零响应、尺度归一化、独立valley符号、多尺度最大值/最佳尺度、峰扫描、区域去重与双门槛、宽度不同于周期、平移不变、端部不可观测、黑带不自动插纸和无效输入拒绝。

运行：`conda run -n pytorch python -m pytest tests/test_v3_shadow.py -q`，以及 `conda run -n pytorch python scripts/v3/run_p1.py`。主输出目录必须不存在，防止覆盖。P0/P1正式完成后的结果、合成报告、真实对照与P2建议见 `P1_RESULTS.md`、`P1_SYNTHETIC_TEST_REPORT.md`、`P1_REAL_SHADOW_COMPARISON.md`、`P2_RECOMMENDATIONS.md`。

所有模块均位于 `paper_counter/v3_shadow/`；旧PaperCounter没有导入它。P1只提取单条中心列作为一维实验，不生成二维中心轨迹、Evidence融合或SheetHypothesis。本轮未执行A5/A6。
