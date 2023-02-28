from collections import deque
import copy
import numpy as np


def ford_fulkerson(graph, graph_ini, source, sink):
    def bfs(start, end):
        q = deque([start])
        visited = set([start])
        while q:
            u = q.popleft()
            for node, cost in graph[u].items():
                if node in visited or cost <= 0:
                    continue

                visited.add(node)
                q.append(node)
                parent[node] = u
                if node == end:
                    return True

        return False

    parent = dict().fromkeys(graph, -1)
    max_flow = 0
    graph_cap = copy.deepcopy(graph)
    graph_flow = copy.deepcopy(graph)
    while bfs(source, sink):
        path_flow = float('inf')
        v = sink
        while v != source:
            path_flow = min(path_flow, graph[parent[v]][v])
            v = parent[v]

        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            if graph[u].get(v):
                graph[u][v] -= path_flow
            else:
                graph[u][v] = -path_flow

            if graph[v].get(u):
                graph[v][u] += path_flow
            else:
                graph[v][u] = path_flow

            v = parent[v]
    # 寻找最小割
    for i in graph_flow:
        for j in graph_flow[i]:
            graph_flow[i][j] = graph_cap[i][j] - graph[i][j]
    s = [source]
    for i in graph_flow:
        if not i in s:
            continue
        for j in graph_flow[i]:
            if j in s:
                continue
            if graph[i][j] != 0:
                s.append(j)
    # print(s)
    # print(list(graph_flow.keys()))
    # s = ['s', 0] # 不调度
    # 计算割的cap
    latency = 0.
    for i in list(s):
        for j in graph_cap[i]:
            if j not in list(s):
                latency += graph_cap[i][j]
    # 计算本地计算时间t_e, 计算服务器计算时间t_s
    t_e = 0.
    t_s = 0.
    for i in range(0, len(graph_cap)-2):
        if i in list(s):
            t_e = t_e + graph_ini[i]['e']
        else:
            t_s = t_s + graph_ini['s'][i]
    # 计算传输时间t_t
    t_t = 0.
    for i in range(len(s)):
        if s[i] == 's':
            continue
        else:
            for j in range(len(graph_ini[s[i]])):
                if (list(graph_ini[s[i]].items())[j][0] not in list(s)) and list(graph_ini[s[i]].items())[j][0] != 'e':
                    t_t = t_t + list(graph_ini[s[i]].items())[j][1]
    return latency, t_e, t_t, t_s
