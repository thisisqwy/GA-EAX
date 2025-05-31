import random
import math
from collections import defaultdict, deque

# Calculate Euclidean distance between two cities
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Total length of a tour
def tour_length(tour, cities):
    return sum(distance(cities[tour[i]], cities[tour[(i+1) % len(tour)]]) for i in range(len(tour)))

# Edge Assembly Crossover (EAX)
def edge_assembly_crossover(parentA, parentB, cities):
    n = len(parentA)
    # === Step 1: Build G_AB ===
    # Each entry G[u] = list of (v, label)
    G = defaultdict(list)
    for tour, lbl in [(parentA, 'A'), (parentB, 'B')]:
        for i in range(n):
            u, v = tour[i], tour[(i+1) % n]
            G[u].append((v, lbl))
            G[v].append((u, lbl))
    # === Step 2: Find AB-cycles ===
    cycles = []
    used = set()
    for start in G:
        for neighbor, lbl in G[start]:
            if (start, neighbor, lbl) in used:
                continue #对于多层循环的continue，会跳过本层循环中本次循环的代码块执行，会跳到本层循环的下一次循环
            cycle = []
            u, current_lbl = start, lbl
            while True:  #在Python中,while True:是创建一个无限循环的常用方法,它表示只要条件为真(True),循环就会一直执行下去、不会自动结束、必须显式地通过某种方式(如break语句)进行中断。
                # find next edge of current_lbl not yet used
                for v, l in G[u]:
                    if l == current_lbl and (u, v, l) not in used:
                        used.add((u, v, l)); used.add((v, u, l))
                        cycle.append((u, v, l))
                        u = v
                        current_lbl = 'B' if current_lbl == 'A' else 'A'
                        break
                else:
                    break  # no continuation
                if u == start and current_lbl == lbl:
                    cycles.append(cycle)
                    break

    # === Step 3: Select E-set of edges ===
    # Example rule: pick the longest AB-cycle
    if not cycles:
        return parentA.copy()
    # choose cycle with max length
    E_cycle = max(cycles, key=lambda c: len(c))
    # collect edges to exchange
    E_A = {(u, v) for u, v, l in E_cycle if l == 'A'}
    E_B = {(u, v) for u, v, l in E_cycle if l == 'B'}

    # === Step 4: Generate intermediate solution ===
    # start from parentA edges
    adj = {i: set() for i in parentA}
    for i in range(n):
        u, v = parentA[i], parentA[(i+1) % n]
        adj[u].add(v); adj[v].add(u)
    # remove edges in E_A, add edges in E_B
    for u, v in E_A:
        adj[u].discard(v); adj[v].discard(u)
    for u, v in E_B:
        adj[u].add(v); adj[v].add(u)

    # === Step 5: Repair subtours ===
    # find connected components
    visited = set()
    subtours = []
    for node in adj:
        if node not in visited:
            comp = []
            queue = deque([node])
            while queue:
                x = queue.popleft()
                if x in visited: continue
                visited.add(x); comp.append(x)
                for y in adj[x]:
                    if y not in visited:
                        queue.append(y)
            subtours.append(comp)
    # connect until one tour remains
    while len(subtours) > 1:
        s1 = subtours.pop(0)
        s2 = subtours.pop(0)
        # find best connection between s1 and s2
        best = None; best_dist = float('inf')
        for a in s1:
            for b in s2:
                d = distance(cities[a], cities[b])
                if d < best_dist:
                    best_dist, best = d, (a, b)
        a, b = best
        # pick arbitrary neighbors to break
        na = next(iter(adj[a]))
        nb = next(iter(adj[b]))
        # remove those edges
        adj[a].remove(na); adj[na].remove(a)
        adj[b].remove(nb); adj[nb].remove(b)
        # add new cross edges
        adj[a].add(b); adj[b].add(a)
        adj[na].add(nb); adj[nb].add(na)
        # merge lists
        subtours.append(s1 + s2)

    # reconstruct tour by walking
    tour = []
    prev = None
    curr = next(iter(adj))
    for _ in range(n):
        tour.append(curr)
        neighbors = [x for x in adj[curr] if x != prev]
        if not neighbors:
            break
        prev, curr = curr, neighbors[0]

    if len(tour) != n:
        return parentA.copy()
    return tour

# Example GA loop
if __name__ == '__main__':
    # random cities
    cities = {i: (random.random(), random.random()) for i in range(30)}
    # initial population
    pop = [random.sample(list(cities.keys()), len(cities)) for _ in range(100)]
    # GA iterations
    for _ in range(200):
        a, b = random.sample(pop, 2)
        child = edge_assembly_crossover(a, b, cities)
        if random.random() < 0.2:
            i, j = random.sample(range(len(child)), 2)
            child[i], child[j] = child[j], child[i]
        pop.sort(key=lambda t: tour_length(t, cities))
        pop[-1] = child
    best = min(pop, key=lambda t: tour_length(t, cities))
    print("Best length:", tour_length(best, cities))
    print(best)
