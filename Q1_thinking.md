
## 问题一
---

### 一、问题一的建模目标

研究固定尺寸圆柱形药材在前 \(1800\ \mathrm s\) 内的：

\[
T(r,t)：\text{温度场},
\]

\[
C(r,t)：\text{干基含水率场}.
\]

题目要求：

- 时间：\(t=100,300,600,900,1200,1500,1800\ \mathrm s\)；
- 距离中心：\(r=0,0.5,1.0,1.5,2.0\ \mathrm{cm}\)；
- 输出完整场：每隔 \(1\ \mathrm s\)、每隔 \(0.1\ \mathrm{cm}\)。

药材尺寸为

\[
R=2\ \mathrm{cm}=0.02\ \mathrm m,
\qquad
L=25\ \mathrm{cm}=0.25\ \mathrm m.
\]

---

### 二、几何简化与基本假设

#### 2.1 轴对称一维假设

由于题目只给出“到药材中心的距离”，且

\[
\frac LR=\frac{25}{2}=12.5,
\]

因此将药材视为轴对称圆柱，只考虑径向变化：

\[
T=T(r,t),\qquad C=C(r,t),
\]

其中

\[
0\le r\le R.
\]

忽略轴向端面效应，相当于认为药材中部区域的径向传递占主导。

#### 2.2 固定尺寸假设

问题一只考虑预热平衡阶段，半径固定为

\[
R=0.02\ \mathrm m.
\]

收缩效应放到问题四处理。

#### 2.3 均匀各向同性假设

假设药材可以看成均匀、各向同性的连续多孔介质，径向导热和水分扩散分别服从 Fourier 定律和 Fick 定律。

#### 2.4 问题一中的参数

题目给出：

\[
\rho=820\ \mathrm{kg/m^3},
\]

\[
c_p=2600\ \mathrm{J/(kg\cdot K)},
\]

\[
k=0.36\ \mathrm{W/(m\cdot K)},
\]

\[
h=25\ \mathrm{W/(m^2\cdot K)},
\]

\[
h_m=8\times10^{-7}\ \mathrm{m/s},
\]

\[
D(C)=7\times10^{-9}
\exp\left(-\frac{0.89}{C}\right)
\ \mathrm{m^2/s}.
\]

初始条件为

\[
T(r,0)=28^\circ\mathrm C,
\]

\[
C(r,0)=2.55.
\]

---

### 三、为什么不能使用集中参数模型

#### 3.1 热 Biot 数

集中参数模型通常要求

\[
Bi=\frac{hL_c}{k}<0.1,
\]

其中

\[
L_c=\frac{V}{A_s}.
\]

对于长圆柱，若只考虑侧面积：

\[
V=\pi R^2L,\qquad A_s=2\pi RL,
\]

所以

\[
L_c=\frac{V}{A_s}=\frac R2=0.01\ \mathrm m.
\]

于是

\[
Bi_h
=
\frac{hL_c}{k}
=
\frac{25\times0.01}{0.36}
\approx0.694.
\]

因此

\[
Bi_h>0.1.
\]

药材内部温度梯度不可忽略，必须求解空间温度场。

\(Bi<0.1\) 是经典传热教材中的工程判据，通常被视为经验规则而非严格定理。参考：[NPTEL Biot number lecture](https://archive.nptel.ac.in/content/storage2/courses/112101001/modules/Module-1/lec7/1.8.html)。

需要区分的是，如果用半径 \(R\) 作径向无量纲化，则会得到

\[
Bi_{h,R}=\frac{hR}{k}\approx1.389.
\]

这个数可用于径向方程分析，但集中参数判据使用的是 \(L_c=V/A_s\)。

#### 3.2 水分 Biot 数

初始含水率下的扩散系数为

\[
D_0
=
7\times10^{-9}
\exp\left(-\frac{0.89}{2.55}\right)
\approx4.94\times10^{-9}\ \mathrm{m^2/s}.
\]

水分 Biot 数为

\[
Bi_m
=
\frac{h_mL_c}{D_0}
=
\frac{8\times10^{-7}\times0.01}
{4.94\times10^{-9}}
\approx1.62.
\]

同样有

\[
Bi_m>0.1.
\]

所以含水率也不能用集中参数处理。

---

### 四、热扩散和水分扩散的时间尺度

热扩散率：

\[
\alpha=\frac{k}{\rho c_p}
=
\frac{0.36}{820\times2600}
\approx1.689\times10^{-7}\ \mathrm{m^2/s}.
\]

热扩散时间尺度：

\[
t_h\sim\frac{R^2}{\alpha}
=
\frac{0.02^2}{1.689\times10^{-7}}
\approx2370\ \mathrm s.
\]

水分扩散时间尺度：

\[
t_m\sim\frac{R^2}{D_0}
=
\frac{0.02^2}{4.94\times10^{-9}}
\approx8.1\times10^4\ \mathrm s.
\]

即：

\[
t_h\approx39.5\ \mathrm{min},
\qquad
t_m\approx22.5\ \mathrm h.
\]

因此在问题一的 \(1800\ \mathrm s\) 内：

- 温度已经开始明显从表面向中心传播；
- 水分扩散相对缓慢；
- 温度场和含水率场都不能简化为均匀值。

---

### 五、温度场控制方程

#### 5.1 微元能量守恒

取半径为 \(r\)、厚度为 \(\mathrm dr\) 的圆柱薄壳。

其体积为

\[
\mathrm dV=2\pi rL\,\mathrm dr.
\]

径向热流密度为

\[
q_r=-k\frac{\partial T}{\partial r}.
\]

能量守恒为：

\[
\text{单位时间内储存的热量}
=
\text{流入热量}-\text{流出热量}.
\]

因此

\[
\rho c_p
\frac{\partial T}{\partial t}
(2\pi rL\,\mathrm dr)
=
2\pi L
\left[
rk\frac{\partial T}{\partial r}
\right]_{r}^{r+\mathrm dr}.
\]

令 \(\mathrm dr\to0\)，得

\[
\rho c_p\frac{\partial T}{\partial t}
=
\frac1r\frac{\partial}{\partial r}
\left(
rk\frac{\partial T}{\partial r}
\right).
\]

所以温度控制方程为

\[
\boxed{
\rho c_pT_t
=
\frac1r\frac{\partial}{\partial r}
\left(rkT_r\right)
}
\]

由于问题一中 \(k\) 为常数，可以进一步写成

\[
\boxed{
T_t
=
\alpha
\left(
T_{rr}+\frac1rT_r
\right)
}
\]

其中

\[
\alpha=\frac{k}{\rho c_p}.
\]

---

### 六、水分场控制方程

#### 6.1 Fick 扩散定律

令水分径向通量为 \(J_r\)，则

\[
J_r=-D(C)\frac{\partial C}{\partial r}.
\]

在轴对称圆柱中，水分守恒方程为

\[
\frac{\partial C}{\partial t}
=
-\frac1r\frac{\partial}{\partial r}(rJ_r).
\]

代入 Fick 定律：

\[
\frac{\partial C}{\partial t}
=
\frac1r\frac{\partial}{\partial r}
\left[
rD(C)\frac{\partial C}{\partial r}
\right].
\]

因此

\[
\boxed{
C_t
=
\frac1r\frac{\partial}{\partial r}
\left[
rD(C)C_r
\right]
}
\]

其中

\[
\boxed{
D(C)=7\times10^{-9}
\exp\left(-\frac{0.89}{C}\right)
}
\]

这个方程是非线性的，因为扩散系数随含水率 \(C\) 变化。

---

### 七、初始条件和边界条件

#### 7.1 圆心对称条件

在圆心 \(r=0\) 处不存在优先方向，所以

\[
\boxed{
T_r(0,t)=0
}
\]

\[
\boxed{
C_r(0,t)=0
}
\]

#### 7.2 表面对流换热条件

药材表面温度记为

\[
T_s(t)=T(R,t).
\]

表面对流热流为

\[
q_s=h[T_s-T_\infty(t)].
\]

内部导热热流为

\[
q_s=-kT_r(R,t).
\]

两者相等，得到

\[
\boxed{
-kT_r(R,t)
=
h[T(R,t)-T_\infty(t)]
}
\]

#### 7.3 表面对流传质条件

药材表面含水率记为

\[
C_s(t)=C(R,t).
\]

表面水分通量满足

\[
J_s=h_m[C_s-C_\infty(t)].
\]

内部水分扩散通量为

\[
J_s=-D(C_s)C_r(R,t).
\]

因此

\[
\boxed{
-D(C_s)C_r(R,t)
=
h_m[C(R,t)-C_\infty(t)]
}
\]

这里暂时把附件一中的烘房水分浓度 \(C_\infty(t)\) 看成有效平衡含水率。

这一点需要后续文献确认。如果附件一的水分浓度实际上是空气湿含量，则应先通过吸附等温线得到

\[
C_e=C_e(T_\infty,\phi_\infty).
\]

#### 7.4 初始条件

\[
\boxed{
T(r,0)=28^\circ\mathrm C
}
\]

\[
\boxed{
C(r,0)=2.55
}
\]

---

### 八、问题一中的热湿场是否解耦

问题一最终的方程组为

\[
T_t
=
\alpha
\left(
T_{rr}+\frac1rT_r
\right),
\]

\[
C_t
=
\frac1r\frac{\partial}{\partial r}
\left[
rD(C)C_r
\right].
\]

可以看到：

- 温度方程不含 \(C\)；
- 水分方程不含 \(T\)；
- 热物性为常数；
- \(D\) 只依赖 \(C\)。

所以在问题一的题面基线下：

\[
\boxed{
\text{温度场和含水率场在数学上完全解耦}
}
\]

可以分别写成

\[
T_t=F_T(T,t),
\]

\[
C_t=F_C(C,t).
\]

但是，真实干燥过程中可能存在：

\[
\text{蒸发潜热耦合},
\]

\[
D=D(C,T),
\]

\[
\text{温度梯度引起的导湿效应}.
\]

任迪峰博士论文中给出了包含潜热项和导湿温项的更完整模型；食品多孔介质干燥研究也通常采用同时传热传质模型。参考：[任迪峰博士论文](<D:/黄洛霖的文件/竞赛/数模赛/CUMUCM2026A/参考文献/中药材干燥过程中质量退化及优化干燥工艺的研究.pdf>)、[Datta 2007](https://www.sciencedirect.com/science/article/abs/pii/S0260877406003980)。

因此本文对问题一的表述应为：

> 由于题面未提供潜热、导湿温系数及温度相关扩散系数，问题一采用题面参数可识别的解耦 Fourier–Fick 基线模型；热湿耦合效应作为后续敏感性分析和模型升级方向。

---

### 九、附件一边界数据处理

附件一给出离散时间点：

\[
(t_j,T_{\infty,j},C_{\infty,j}).
\]

对于

\[
t_j\le t\le t_{j+1},
\]

采用分段线性插值：

\[
T_\infty(t)
=
T_{\infty,j}
+
\frac{t-t_j}{t_{j+1}-t_j}
\left(
T_{\infty,j+1}-T_{\infty,j}
\right),
\]

\[
C_\infty(t)
=
C_{\infty,j}
+
\frac{t-t_j}{t_{j+1}-t_j}
\left(
C_{\infty,j+1}-C_{\infty,j}
\right).
\]

这样做的原因是：

- 附件一数据本身按时间离散给出；
- 分段线性插值不会制造新的极值；
- 避免高阶多项式插值造成温度或湿度越界。

---

### 十、有限体积离散

#### 10.1 径向网格

将半径划分为

\[
0=r_0<r_1<\cdots<r_N=R,
\]

\[
\Delta r=\frac RN.
\]

输出要求为 \(0.1\ \mathrm{cm}\)，因此最终可取

\[
\Delta r=0.001\ \mathrm m.
\]

时间步长最终取

\[
\Delta t=1\ \mathrm s.
\]

#### 10.2 控制体

第 \(i\) 个节点的控制体边界记为

\[
r_{i-\frac12},\qquad r_{i+\frac12}.
\]

控制体的径向面积因子为

\[
V_i'
=
\frac12
\left(
r_{i+\frac12}^2-r_{i-\frac12}^2
\right).
\]

实际控制体体积为

\[
V_i=2\pi L V_i'.
\]

统一的 \(2\pi L\) 在方程两边可以约去，因此计算时可使用 \(V_i'\)。

---

### 十一、温度方程离散

定义界面热通量因子

\[
F_{i+\frac12}^{T}
=
r_{i+\frac12}k
\frac{T_{i+1}-T_i}{\Delta r}.
\]

对控制体积分，并采用后向 Euler 离散：

\[
\boxed{
\rho c_pV_i'
\frac{T_i^{n+1}-T_i^n}{\Delta t}
=
F_{i+\frac12}^{T,n+1}
-
F_{i-\frac12}^{T,n+1}
}
\]

其中 \(n\) 表示当前时间层。

#### 11.1 圆心控制体

圆心处由于对称性：

\[
F_{-\frac12}^{T}=0.
\]

因此圆心控制体不需要额外设置虚拟边界。

#### 11.2 表面控制体

表面通量由 Robin 边界直接给出：

\[
-kT_r(R,t)=h(T_N-T_\infty).
\]

由于定义

\[
F^T=RkT_r,
\]

所以

\[
\boxed{
F_{N+\frac12}^{T}
=
-Rh(T_N-T_\infty)
}
\]

当 \(T_N<T_\infty\) 时，该项为正，表示热量进入药材。

由于 \(\rho,c_p,k\) 均为常数，温度方程每个时间步形成一个三对角线性方程组。

---

### 十二、水分方程离散

定义界面水分通量因子：

\[
F_{i+\frac12}^{C}
=
r_{i+\frac12}
D_{i+\frac12}
\frac{C_{i+1}-C_i}{\Delta r}.
\]

为了保证不同含水率区域之间的通量连续，扩散系数采用调和平均：

\[
D_{i+\frac12}
=
\frac{2D_iD_{i+1}}{D_i+D_{i+1}}.
\]

离散方程为

\[
\boxed{
V_i'
\frac{C_i^{n+1}-C_i^n}{\Delta t}
=
F_{i+\frac12}^{C,n+1}
-
F_{i-\frac12}^{C,n+1}
}
\]

#### 12.1 圆心边界

\[
F_{-\frac12}^{C}=0.
\]

#### 12.2 表面边界

由

\[
-D(C_s)C_r(R,t)
=
h_m(C_N-C_\infty),
\]

得到

\[
\boxed{
F_{N+\frac12}^{C}
=
-Rh_m(C_N-C_\infty)
}
\]

当

\[
C_N>C_\infty
\]

时，

\[
F_{N+\frac12}^{C}<0,
\]

表示水分从药材内部流出。

---

### 十三、非线性水分方程的迭代

因为

\[
D_i=7\times10^{-9}\exp(-0.89/C_i),
\]

扩散系数随未知量 \(C_i\) 变化，因此每个时间步需要迭代。

第 \(n+1\) 个时间层的计算过程为：

1. 取初始猜测

\[
C_i^{(0)}=C_i^n.
\]

2. 根据第 \(m\) 次迭代结果计算

\[
D_i^{(m)}
=
7\times10^{-9}
\exp\left(-\frac{0.89}{C_i^{(m)}}\right).
\]

3. 计算界面扩散系数 \(D_{i+1/2}^{(m)}\)。

4. 固定扩散系数，求解新的线性方程组，得到

\[
C_i^{(m+1)}.
\]

5. 判断是否满足

\[
\max_i
\frac{|C_i^{(m+1)}-C_i^{(m)}|}
{1+|C_i^{(m+1)}|}
<\varepsilon.
\]

若满足，则令

\[
C_i^{n+1}=C_i^{(m+1)}.
\]

否则继续迭代。

---

### 十四、无量纲参数总结

热 Fourier 数：

\[
Fo_h=\frac{\alpha t}{R^2}.
\]

在 \(t=1800\ \mathrm s\) 时：

\[
Fo_h\approx0.760.
\]

水分 Fourier 数：

\[
Fo_m=\frac{D_0t}{R^2}
\approx0.022.
\]

这说明问题一中热扩散明显快于水分扩散。

因此可以预期：

\[
T_{\text{表面}}>T_{\text{中心}},
\]

\[
C_{\text{中心}}>C_{\text{表面}}.
\]

---

### 十五、整体守恒验证

#### 15.1 热量守恒

定义药材总显热：

\[
E(t)
=
2\pi L
\int_0^R
\rho c_pT(r,t)r\,\mathrm dr.
\]

则理论上应满足

\[
\boxed{
\frac{\mathrm dE}{\mathrm dt}
=
2\pi RLh[T_\infty(t)-T_s(t)]
}
\]

数值计算中应检查左、右两侧的累计误差。

#### 15.2 水分守恒

在当前有效扩散模型下，定义归一化水分总量：

\[
M(t)
=
2\pi L
\int_0^R C(r,t)r\,\mathrm dr.
\]

理论上有

\[
\boxed{
\frac{\mathrm dM}{\mathrm dt}
=
-2\pi RLh_m[C_s(t)-C_\infty(t)]
}
\]

如果误差不能随着网格加密而下降，说明边界符号、径向面积因子或离散格式存在问题。

---

### 十六、问题一的最终模型

综合以上内容，问题一的初边值问题为：

\[
\boxed{
\rho c_p\frac{\partial T}{\partial t}
=
\frac1r\frac{\partial}{\partial r}
\left(
rk\frac{\partial T}{\partial r}
\right)
}
\]

\[
\boxed{
\frac{\partial C}{\partial t}
=
\frac1r\frac{\partial}{\partial r}
\left[
rD(C)\frac{\partial C}{\partial r}
\right]
}
\]

其中

\[
D(C)=7\times10^{-9}
\exp\left(-\frac{0.89}{C}\right).
\]

边界条件：

\[
T_r(0,t)=0,\qquad C_r(0,t)=0,
\]

\[
-kT_r(R,t)
=
h[T(R,t)-T_\infty(t)],
\]

\[
-D(C_s)C_r(R,t)
=
h_m[C(R,t)-C_\infty(t)].
\]

初始条件：

\[
T(r,0)=28^\circ\mathrm C,
\]

\[
C(r,0)=2.55.
\]

最终采用：

\[
\boxed{
\text{一维径向固定域}
+
\text{Fourier导热}
+
\text{Fick非线性扩散}
+
\text{Robin边界}
+
\text{数学解耦}
+
\text{有限体积离散}
}
\]

