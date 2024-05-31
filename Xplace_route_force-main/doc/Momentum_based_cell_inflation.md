# Momentum-based cell inflation

<img src="Xplace_cell_inflate_formula.png" alt="alt text" width="50%">

$cg^0_{i,t} = \sum\limits_{(x,y) \in R_R \times R_c, A_i \cap A_{x,y} \neq \varnothing} \frac{A_i \cap A_{x,y}}{A_i} \cdot IR_{x,y}$


- 令$cg_{i,t}=\sqrt{cg^0_{i,t}} - 1.0$，其中 $cg_{i,t}^{0.0}$ 表示 $cg_{i,t}$ 中大于0.0的元素，即位于拥塞区域的单元的膨胀率（Xplace的）。
- $r_{i,t}$ 表示 $cell_i$ 在第 $t$ 次单元膨胀优化迭代中的宽度和高度的膨胀率。
- $\Delta r_{i,t}$ 表示第 $t$ 次迭代中 $cell_i$ 的膨胀比率相对于上一次迭代的变化。
- $c_{i,t}$ 表示第 $t$ 次迭代中 $cell_i$ 的膨胀率的关键度（类比梯度，需要弄出负值表示缩小）（是关键）。

可以使用以下公式来计算相关的值：

- $r_{i,t+1} = r_{i,t} + \Delta r_{i,t}，（r_{i,0} = I^{n \times 2}，即膨胀率1.0）$
- $\Delta r_{i,t+1} = 0.5 \Delta r_{i,t} + 0.5 c_{i,t}，（\Delta r_{i,0} = c_{i,0}）$
- 暂定$c_{i,t} =  \begin{cases} 
              cg_{i,0} & \text{if } t = 0 \\
              cg_{i,t}^{0.0} - cg_{i,t}^{0.0}.mean() & \text{if } t \neq 0 
              \end{cases}$
见函数 cal_cell_inflation_criticality
