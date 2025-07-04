import numpy as np
import random
import math
from collections import defaultdict, deque
import time
start_time = time.time()


class eax_tabu():
    def __init__(self,num_pop,num_city,noff,data):
        self.num_pop = num_pop
        self.num_city = num_city
        self.noff=noff
        self.location = data
        self.dis_mat = self.compute_dismat()
        self.pop = self.init_pop()
        self.fitness=self.compute_fitness(self.pop)
    #根据二维坐标计算距离矩阵
    def compute_dismat(self):
        matrix=np.zeros((self.num_city,self.num_city))
        for i in range(self.num_city):
            for j in range(self.num_city):
                if i==j:
                    matrix[i,j]=np.inf
                else:
                    matrix[i,j]=math.sqrt((self.location[i][0]-self.location[j][0])**2+(self.location[i][1]-self.location[j][1])**2)
        return matrix
    #种群初始化，先随机生成解，之后对于每个解都2-opt算法优化。
    def init_pop(self):
        pop=[]
        for i in range(self.num_pop):
            pop.append(np.array(random.sample(range(self.num_city),self.num_city))) #用数组表示路径，更高效。
        for i in range(self.num_pop):#对初试解的每个解都用2-opt算法优化到无法再优化的地步。
            improved = True
            best=pop[i].copy()
            best_dist = self.path_length(best)
            while improved:
                improved = False
                for j in range(1,self.num_city-1):
                    for k in range(j+1,self.num_city):
                        new = self.dis_mat[best[j-1],best[k]]+self.dis_mat[best[j],best[(k+1)%self.num_city]]
                        old = self.dis_mat[best[j-1],best[j]]+self.dis_mat[best[k],best[(k+1)%self.num_city]]
                        if new < old:
                            best[j:k + 1] = best[j:k + 1][::-1]  # 如果k+1=num_city，那么best[k+1:]就会等于一个空列表。
                            best_dist = best_dist - old + new
                            improved = True
                            break  # 退出内层循环
                    if improved:
                        break  # 退出外层循环
            pop[i]=best.copy()
        return pop
    #计算单条路径的长度
    def path_length(self,path):
        sum=self.dis_mat[path[-1]][path[0]]
        for i in range(len(path)-1):
            sum=sum+self.dis_mat[path[i]][path[i+1]]
        return sum
    #计算种群适应度函数
    def compute_fitness(self,fruits):#适应度函数为路径长度的倒数。
        score=[]
        for fruit in fruits:
            length = self.path_length(fruit)
            score.append(1.0/length)
        return np.array(score)#np.array将score转化为数组的同时，其实也是创造了一个score的副本。
    #将一个解进行标准化
    def canonical_path(self,path):
        # 找到最小节点在 path 中的所有位置（通常只有一个）
        min_node = min(path)
        # 找到第一个最小节点的位置
        idx = np.where(path == min_node)[0][0]
        # 从 idx 开始循环移位
        rotated = np.concatenate((path[idx:], path[:idx]))
        return rotated
    #判断两个解是否一样
    def not_equal(self,path1,path2):
        #先将path2翻转，看下是否一样
        reversed_path2=path2[::-1]
        not_equal=not np.array_equal(path1,reversed_path2)
        if not_equal:
            new_path1=self.canonical_path(path1)
            new_path2=self.canonical_path(path2)
            not_equal = not np.array_equal(new_path1, new_path2)
        return not_equal
    def get_edges(self,tour):
        n=len(tour)
        return [frozenset({tour[i], tour[(i+1) % n]})#用frozenset表示每条边，无向边。
                for i in range(n)]
    def has_any_edge(self,E):
        return any(E[v]['A'] or E[v]['B'] for v in E)
    #将A与B的边的并集分解成AB-cycles
    def decompose_AB_cycles(self, EA, EB):
        # 将 EA, EB 转为 set 以加速
        EA_set = set(EA)
        EB_set = set(EB)
        E=defaultdict(lambda: {'A': set(), 'B': set()})#E是两个父代的边的集合GAB
        for e in EA_set: #把EA和EB两个边集的无向边加进GAB中
            u, v = tuple(e)
            E[u]['A'].add(e)
            E[v]['A'].add(e)
        for e in EB_set:
            u, v = tuple(e)
            E[u]['B'].add(e)
            E[v]['B'].add(e)
        ab_cycles = []
        while self.has_any_edge(E):#当整个E中一条边都没有时就表示找到了全部的AB-cycle
            start_node = random.choice([v for v in E if E[v]['A'] or E[v]['B']])
            start_label=random.choice(['A', 'B'])
            curr_node=start_node
            curr_label=start_label
            traced_path=[]
            visited_nodes=[]#记录已访问的点，每添加一条边，便标记这条边的起点，即curr_node
            while True:
                cand_edges= list(E[curr_node][curr_label])#备选边是在当前节点和当前标签下可选的下一条边。
                e = random.choice(cand_edges)
                u, v = tuple(e)
                next_node=v if curr_node == u else u
                next_label= 'B' if curr_label=='A' else 'A'
                for x in (u, v):
                    E[x][curr_label].remove(e)
                if next_node in visited_nodes and next_label==traced_path[visited_nodes.index(next_node)][0]:
                    idx=visited_nodes.index(next_node)
                    cycle=traced_path[idx:]
                    cycle.append((curr_label,e))
                    ab_cycles.append(cycle)
                    traced_path = traced_path[:idx]
                    visited_nodes=visited_nodes[0:idx]
                    curr_node=next_node
                    curr_label=next_label

                else:
                    traced_path.append((curr_label,e))
                    visited_nodes.append(curr_node)
                    curr_node = next_node
                    curr_label = next_label
                if  not traced_path:#traced_path为空时退出while循环，
                    break
        return ab_cycles
    def get_feasible(self,AB_cycles):
        feasible=[
            cycle
            for cycle in AB_cycles
            # 排除长度为 2 的
            if len(cycle) != 2]
        return feasible
    def find_Eset(self,cycles):
        Eset=[]
        if cycles:#这里做判断cycles是否为空，如果为空就不做下面的操作了，节省时间。
            for cycle in cycles:
                if random.random()<0.5:
                    Eset.append(cycle)
        return Eset
    def generate_child(self,EA,Eset):
        #先生成后代的边集
        EC=EA.copy()
        for cycle in Eset:
            for lb,e in cycle:
                if lb == 'A':
                    EC.remove(e)
                else:  # lb == 'B'
                    EC.append(e)
        # 提取所有subtour形成一个中间解，只有当subtours中只有一个1个subtour时才代表是可行的。
        subtours=self.extract_subtours(EC)
        #若subtour数量大于1，则需要通过循环不断合并subtour来得到最终的可行解子代。
        while len(subtours) > 1:
            subtours.sort(key=len)
            U = subtours[0]
            best = float('inf')
            best_result = None
            for i in range(1, len(subtours)):
                V = subtours[i]
                reconnect = self.reconnect_two(U, V)
                if reconnect is None:
                    continue
                (del_u, del_v), (add_u, add_v) = reconnect
                delta = (self.dis_mat[list(add_u)[0]][list(add_u)[1]] + self.dis_mat[list(add_v)[0]][list(add_v)[1]]) - \
                        (self.dis_mat[list(del_u)[0]][list(del_u)[1]] + self.dis_mat[list(del_v)[0]][list(del_v)[1]])
                if delta < best:
                    best = delta
                    best_result = (i, reconnect)

            if best_result is None:
                break
            idx, ((del_u, del_v), (add_u, add_v)) = best_result
            V = subtours[idx]
            for e in (del_u, del_v):
                if e in EC:
                    EC.remove(e)
            for e in (add_u, add_v):
                if e not in EC:
                    EC.append(e)
            subtours = self.extract_subtours(EC)
        return subtours[0]
    def extract_subtours(self,EC_edges):
        adj = defaultdict(list)
        for e in EC_edges:
            u, v = tuple(e)
            adj[u].append(v)
            adj[v].append(u)
        visited = set()
        tours = []
        for node in adj: #在最终的中间解中，每个顶点代表的城市都只存在于一个subtour里，所以node被遍历到了就会被visited
            if node in visited:
                continue
            tour = []
            curr, prev = node, None
            while True:
                tour.append(curr)
                visited.add(curr)
                nbrs = [w for w in adj[curr] if w != prev]
                if not nbrs:#对于那种在subtours中只有两个点的subtour，这里的if not nbrs是必要的。
                    break
                prev, curr = curr, nbrs[0]#这时候nbrs肯定有两个点，分别代表curr这个点两个不同方向的临近点，随便取一个即可。
                if curr == node:
                    break
            tours.append(tour)
        return tours
    def reconnect_two(self,tour_u, tour_v):
        """按最小增量策略合并两个子巡回，返回要删除和添加的边对。"""
        best_delta = float('inf')
        best = None
        for i in range(len(tour_u)):
            u1, u2 = tour_u[i], tour_u[(i + 1) % len(tour_u)]
            w_u = self.dis_mat[u1][u2]
            for j in range(len(tour_v)):
                v1, v2 = tour_v[j], tour_v[(j + 1) % len(tour_v)]
                w_v = self.dis_mat[v1][v2]
                d_u2_v1 = self.dis_mat[u2][v1]
                d_u1_v2 = self.dis_mat[u1][v2]
                delta = -w_u - w_v + d_u2_v1 + d_u1_v2
                if delta < best_delta:
                    best_delta = delta
                    best = ((frozenset({u1, u2}), frozenset({v1, v2})),
                            (frozenset({u2, v1}), frozenset({u1, v2})))
        return best
    def main(self):
        counter=0
        sort_index = np.argsort(-self.fitness).copy()
        best_path = self.pop[sort_index[0]].copy()
        best_fitness =self.fitness[sort_index[0]]
        print(f"初始解的最优路径为:{best_path}")
        print('初始解的最短路径长度为：%.2f' % (1/best_fitness))
        while counter<30 and self.not_equal(self.pop[0],self.pop[1]):
            counter+=1  #每迭代一次，counter就加1
            new_population = [None for _ in range(self.num_pop)]#创建了两个空的大列表
            #每代进行EAX前都要生成一个随机排列
            r=random.sample(range(0, self.num_pop), self.num_pop)
            edges=[]#edges是边集合的一个列表集合
            for i in range(self.num_pop):#edges中的顺序跟r是一样的，即edges中的i和ri是对应的
                edges.append(self.get_edges(self.pop[r[i]]))
            #配对，生成300对父母。这里采取循环一对对父母进行EAX
            for i in range(self.num_pop):
                Pa = self.pop[r[i]].copy()
                Pb = self.pop[r[(i + 1) % self.num_pop]].copy()
                best_length=self.path_length(Pa)
                child_noff=[None for _ in range(self.noff)]
                child_Eset=[None for _ in range(self.noff)]
                #通过分解Pa和Pb的边的并集得到AB-cycles
                AB_cycles = self.decompose_AB_cycles(edges[i], edges[(i+1)%self.num_pop])
                # 接下来生成Noff个后代，即进行Noff次循环，
                for j in range(self.noff):
                    # 删除那些无效的AB-cycle(即边的数量为2的AB-cycle)
                    feasible_AB_cycles=self.get_feasible(AB_cycles)
                    # 从新的AB-cycle集合中采用随机策略来选择有效的AB-cycle构建一个E-set。
                    child_Eset[j]=self.find_Eset(feasible_AB_cycles)
                    #如果构建的E-set是空的，那么后代就直接等于父代Pa
                    if not child_Eset[j]: #如果用随机策略去生成Eset，结果一个AB-cycle都没选到，那么子代就直接等于父代Pa
                        child_noff[j]=self.pop[r[i]].copy()
                    else:
                        child_noff[j]=self.generate_child(edges[i],child_Eset[j])#用Pa的边集和上面得到的Eset生成一个可行解作为后代。
                #minindex_child等于路径长度最短的那个后代在child中的索引。
                minindex_child=min(range(len(child_noff)), key=lambda k: self.path_length(child_noff[k]))
                if self.path_length(child_noff[minindex_child]) < best_length:#这里一定要取小于号。如果取等于号，可能会把空的Eset添加进pre-Eset。
                    new_population[r[i]]=child_noff[minindex_child].copy()
                else:
                    new_population[r[i]]=self.pop[r[i]].copy()
            self.pop=new_population.copy()
            #重新计算这一代的fitness
            self.fitness = self.compute_fitness(self.pop)
            new_sort_index = np.argsort(-self.fitness).copy()#找出
            new_best_path = self.pop[sort_index[0]].copy()
            new_best_fitness = self.fitness[sort_index[0]]
            if new_best_fitness>best_fitness:
                counter=0
                best_path = new_best_path.copy()
                best_fitness=new_best_fitness
            print(f"当前的最优路径为:{best_path}")
            print('当前的最短路径长度为：%.2f' % (1 / best_fitness))
# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data
data = read_tsp(r'D:\Users\qwy\Desktop\tsp算例\st70.tsp\st70.tsp')
data = np.array(data)
data = data[:, 1:]
model=eax_tabu(num_pop=300,num_city=data.shape[0],noff=50,data=data.copy())
model.main()
end_time = time.time()
print("代码运行时间：", end_time - start_time, "秒")