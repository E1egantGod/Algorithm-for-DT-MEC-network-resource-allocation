import torch
import torch.nn as nn
import numpy as np
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from DNNDAG import getAlex, getVGG19, getResNet
from ford_fulkerson import ford_fulkerson
import gym
import math, os
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.025
MAX_EP = 10
MAX_EP_STEP = 30

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = 2 * torch.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(-1, ).data, sigma.view(-1, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, gains, glob_powers, glo_gains, n_ED):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.powers = glob_powers
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.gains = gains
        self.no = name
        self.g_gains = glo_gains[:]
        self.n_ED = n_ED

    def run(self):
        p_max = 10
        while self.g_ep.value < MAX_EP:
            total_step = 0
            s = np.array(self.gains).reshape(self.n_ED)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            self.t_enq = np.zeros(self.n_ED) # 本地计算队列结束时间点
            self.t_trq = np.zeros(self.n_ED) # 传输队列结束时间点
            self.t_seq = np.zeros(self.n_ED) # 服务器计算队列结束时间点
            self.t_wait = np.zeros(self.n_ED)
            self.t_start = np.zeros(self.n_ED)
            self.t_tr_q = np.zeros(self.n_ED)
            self.t_se_q = np.zeros(self.n_ED)
            self.t_task = 0.1 # 本地计算任务到达时间间隔长度
            for t in range(MAX_EP_STEP):
                done = False
                a = self.lnet.choose_action(v_wrap(s[None, :])).clip(0.5,p_max)
                if self.name == 'w0':
                    self.powers[0:self.n_ED] = a
                elif self.name == 'w1':
                    self.powers[self.n_ED:self.n_ED*2] = a
                else:
                    self.powers[self.n_ED*2:self.n_ED*3] = a

                r = self.step(a,t)
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)    # normalize
                # if self.name == 'w0':
                #     print(t,r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                total_step += 1

        self.res_queue.put(None)

    def step(self, a, task_no):
        s_ = self.gains
        BW = 1e6*3/self.n_ED
        # self.powers[:]
        rate = np.zeros(self.n_ED)
        delay = np.zeros(self.n_ED)

        self.t_arrive = task_no * self.t_task

        for i in range(self.n_ED):
            P = self.gains[i]*a[i]
            PI = 0.
            PI = self.g_gains[3*self.n_ED*self.no + i]*self.powers[:][i] + self.g_gains[3*self.n_ED*self.no + i+self.n_ED]*self.powers[:][i+self.n_ED] + self.g_gains[3*self.n_ED*self.no + i+self.n_ED*2]*self.powers[:][i+self.n_ED*2]
            I = PI - P
            N = 1e-22
            rate[i] = BW * math.log2(1+P/(I+N))
            if self.t_arrive < self.t_enq[i]: # 到达早于本地计算队列结束，需要排队
                self.t_wait[i] = self.t_enq[i] - self.t_arrive
                self.t_start[i] = self.t_enq[i]
            else: # 不需要排队
                self.t_wait[i] = 0
                self.t_start[i] = self.t_arrive
            self.t_tr_q[i] = max(self.t_trq[i] - self.t_start[i], 0)
            self.t_se_q[i] = max(self.t_seq[i] - self.t_start[i], 0)
            DAG, DAG_ini = getAlex(rate[i], self.t_tr_q[i], self.t_se_q[i])
            latency, t_e, t_t, t_s = ford_fulkerson(DAG, DAG_ini, 's', 'e')
            delay[i] = latency + self.t_wait[i]
            # print(self.no,"         ",delay[i])
            self.t_enq[i] = self.t_start[i] + t_e

            if self.t_trq[i] < self.t_enq[i]:
                self.t_trq[i] = self.t_enq[i] + t_t
            else:
                self.t_trq[i] += t_t

            if self.t_seq[i] < self.t_trq[i]:
                self.t_seq[i] = self.t_trq[i] + t_s
            else:
                self.t_seq[i] += t_s
        r = -np.array(delay).mean()
        return r

def channel_gain(distance, shape):
    path_loss_bs_user = 37 + 30 * np.log2(distance)
    path_loss_bs_user = path_loss_bs_user + generate_shadow_fading(0, 8, shape, 1)
    gain = np.power(10, - path_loss_bs_user / 20)
    return gain

def generate_shadow_fading(mean, sigma, num_user, num_bs):
    sigma = np.power(10, sigma/10)
    mean = np.power(10, mean/10)
    m = np.log(mean**2/np.sqrt(sigma**2 + mean**2))
    sigma = np.sqrt(np.log(sigma**2/mean**2 + 1))
    lognormal_fade = np.exp(np.random.randn(num_user, num_bs) * sigma + m)
    return lognormal_fade.reshape(num_user)

if __name__ == "__main__":
    f = open("r.txt",'w').close()
    # 初始化环境 1个DTS-global n个SBS-worker 每个SBS关联m个ED
    n_worker = 3
    loc_worker = [[0,0],[570,350],[570,-350]]
    r_worker = 100
    n_ED = 10
    loc_ED = []
    glob_powers = mp.Array('d',[0. for x in range(n_worker*n_ED)])

    # 随机放置位置,worker1对应ED1-3,worker2对应ED4-6,worker3对应ED7-9
    for i in range(n_worker): 
        dis = []
        for j in range(n_ED):
            loc_ED.append(np.random.rand(2)*r_worker + loc_worker[i])

    # 计算全局gain
    g_gain = []
    for i in range(n_worker):
        dis = []
        for j in range(n_ED*n_worker):
            dis.append(math.sqrt((loc_ED[j][0] - loc_worker[i][0])**2 + (loc_ED[j][1] - loc_worker[i][1])**2))
        g_gain.extend(channel_gain(dis,n_ED*n_worker))
        # print(dis)
    glo_gains = mp.Array('d',g_gain)

    # 提取状态gain
    gain_state = []
    # gain_state.append(g_gain[0:3])
    # gain_state.append(g_gain[12:15])
    # gain_state.append(g_gain[24:27])
    gain_state.append(g_gain[0:n_ED])
    gain_state.append(g_gain[n_ED*(n_worker+1):n_ED*(n_worker+2)])
    gain_state.append(g_gain[n_ED*(2*n_worker+2):n_ED*(2*n_worker+3)])
    # print(gain_state)

    N_S = n_ED # 3个ED的信道增益
    N_A = n_ED # 3个ED的功率

    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, gain_state[i], glob_powers, glo_gains, n_ED) for i in range(n_worker)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Episodes')
    # plt.show()
