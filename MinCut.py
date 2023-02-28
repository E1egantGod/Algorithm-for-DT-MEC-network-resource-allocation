import numpy as np
from DNNDAG import getAlex, getVGG19, getResNet
from ford_fulkerson import ford_fulkerson


# 传输速率 CAT1，3G，   4G，    5G
rate = [1.3e5, 1.1e6, 5,85e6, 1e8]

# 队列长度
t_enq = 0 # 本地计算队列结束时间点
t_trq = 0 # 传输队列结束时间点
t_seq = 0 # 服务器计算队列结束时间点
t_task = 0.0001 # 本地计算任务到达时间间隔长度

n_task_max = 20
latencys = []
reward_o_data = open("reward.txt",'w')
for i in range (0, n_task_max):
	# print(i)
	t_arrive = i * t_task

	if t_arrive < t_enq: # 到达早于本地计算队列结束，需要排队
		t_wait = t_enq - t_arrive
		t_start = t_enq
	else: # 不需要排队
		t_wait = 0
		t_start = t_arrive
	# print(t_wait)
	t_tr_q = max(t_trq - t_start, 0)
	t_se_q = max(t_seq - t_start, 0)

	# 获得DNN的时延图（考虑调度队列）
	DAG, DAG_ini = getAlex(rate[3], t_tr_q, t_se_q)

	# 根据DNN的时延图进行最小割
	latency, t_e, t_t, t_s = ford_fulkerson(DAG, DAG_ini, 's', 'e')
	latency += t_wait
	# print(t_e,t_t,t_s)
	latencys.append(latency)
	reward_o_data.write(str(latency)+"\n")
	# print("Task ", i, ", latency: ",latency , t_tr_q, t_se_q)
	print("Task ", i, ", latency: ",latency)

	# 排队信息更新
	t_enq = t_start + t_e

	if t_trq < t_enq:
		t_trq = t_enq + t_t
	else:
		t_trq += t_t
	if t_seq < t_trq:
		t_seq = t_trq + t_s
	else:
		t_seq += t_s