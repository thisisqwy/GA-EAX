# GA-EAX
A genetic algorithm using Edge Assembly Crossover for the traveling salesman problem(Accompanied by Chinese annotations)  
EAX_Rand：算例是st70，官方最优值为675，参数设置：种群数量记为100，每对父代产生子代数 Noff = 50，种群初始化为随机生成。终止条件是若整个种群收敛到同一解，或连续 30 代无全局最优更新，则停止  
EAX_Single:除了选择AB-cycle是只选一个外，其他一样。  
EAX_TabuScheme：在EAX-Rand的基础上加了一个禁忌边集，包含禁忌边的AB-cycle无法被选择。每个个体保存的禁忌边集上限为3，即tenure=3
