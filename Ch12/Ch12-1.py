import numpy as np
import pandas as pd
import time

np.random.seed(1)

# 寻找路径的长度是6
N_STATES = 6
# 有两个动作，要么左，要么右
ACTIONS = ['left', 'right']
# 0.9的情况下，按照Q表决定方向
EPSILON = 1.0
# 学习率
ALPHA = 0.1
# 眼光长远
GAMMA = 0.9
# 最大迭代次数
MAX_EPISODES = 10
# 刷新时间
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        # 初始化一个全为0的Q表
        np.zeros((n_states, len(actions))),
        # 列名是动作名
        columns=actions,
    )
    return table


# 选择动作
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        # 随机选择
        action_name = np.random.choice(ACTIONS)
    else:
        # 按照Q表选择
        action_name = state_actions.idxmax()
    return action_name


# 获取环境的反馈
def get_env_feedback(S, A):
    # 向右
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    # 向左
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


# 更新环境
def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


# 强化学习
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            print('选择：', A)
            # 采取动作，获得反馈
            S_, R = get_env_feedback(S, A)
            print('反馈：', S_, R)
            q_predict = q_table.loc[S, A]
            print('q_predict = q_table.loc[S, A]：')
            print(q_table)
            print(q_predict)
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
                print('S_ != terminal：', q_target)
            else:
                q_target = R
                print('terminal：', q_target)
                is_terminated = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            print(q_table)
            S = S_
            print('S = S_', S_, S)

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
