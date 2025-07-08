# GA-EAX
A genetic algorithm using Edge Assembly Crossover for the traveling salesman problem(Accompanied by Chinese annotations)  
EAX_Rand：算例是st70，官方最优值为675，参数设置：种群数量记为100，每对父代产生子代数 Noff = 50，种群初始化为随机生成。终止条件是若整个种群收敛到同一解，或连续 30 代无全局最优更新，则停止  
EAX_Single:除了选择AB-cycle是只选一个外，其他一样。  
EAX_TabuScheme：在EAX-Rand的基础上加了一个禁忌边集，包含禁忌边的AB-cycle无法被选择。每个个体保存的禁忌边集上限为3，即tenure=3。  
本文遗传算法步骤：
步骤1：初始化，随机生成Npop条路径，之后对于每条路径都采用2-opt进行一个优化。
步骤2：生成一个关于数字1,2,...,Npop的随机排列r。生成Npop对父母，分别是(Pr1,Pr2),...(PrNpop,Pr1).
步骤3：对每对父代 {pᵢ,pᵢ₊₁}，执行noff次EAX算子，产生子代集合 {c₁,…,c_Noff}。
步骤4：在 {c₁,…,cNoff} ∪ {pᵢ} 中选出路径长度最短的那一条 cbest，替换 pᵢ，
步骤5：若整个种群收敛到同一解，或连续 30 代无全局最优更新，则停止；否则转第 2 步  
EAX_tabu:
步骤1：记父代Pa和父代Pb，记Gab为Pa和Pb边的并集。
步骤2：将Gab分解为一个关于AB-cycle的集合。
步骤3：从父代Pa的pre-Eset中随机选择一部分边，即random strategy，每条边被选中的概率依旧是0.5。这些边的集合记为tabu-edges。
步骤4：在步骤2得到的AB-cycle集合中删除那些包含了tabu-edge的AB-cycle。
步骤5：从新的AB-cycle集合中采用随机策略来选择有效的AB-cycle构建一个E-set。
步骤6：在Pa中删除E-set里的A边，并添加E-set里的B边，从而形成一个中间解，可能有若干subtour，也可能正好是一个可行解。
步骤7：通过2-opt算法每次将中间解的两个subtour连接，重复进行直到只剩一个tour。同时在Pa的pre-Eset中添加步骤5得到的E-set，并删除pre-Eset中最老的那个。

