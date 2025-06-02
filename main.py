import random
import numpy as np
import torch
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot  # PyTorch版本
import matplotlib.pyplot as plt
from Maze import Maze
import time
from Runner import Runner

def my_search(maze):
    """
    使用深度优先搜索(DFS)算法解决迷宫问题
    Args:
        maze: 迷宫对象，包含迷宫的结构和状态信息
    Returns:
        path: 从起点到终点的路径列表
    """
    # 定义机器人移动方向映射
    move_map = {
        'u': (-1, 0),  # 向上移动
        'r': (0, +1),  # 向右移动
        'd': (+1, 0),  # 向下移动
        'l': (0, -1),  # 向左移动
    }

    class SearchTree(object):
        """
        搜索树节点类，用于构建DFS搜索树
        每个节点代表迷宫中的一个位置，并记录到达该位置的路径信息
        """
        def __init__(self, loc=(), action='', parent=None):
            self.loc = loc  # 当前节点位置
            self.to_this_action = action  # 到达当前节点的动作
            self.parent = parent  # 当前节点的父节点
            self.children = []  # 当前节点的子节点列表

        def add_child(self, child):
            self.children.append(child)

        def is_leaf(self):
            return len(self.children) == 0

    def expand(maze, is_visit_m, node):
        """
        扩展当前节点，生成所有可能的子节点
        Args:
            maze: 迷宫对象
            is_visit_m: 访问标记矩阵
            node: 待扩展的节点
        Returns:
            int: 新生成的子节点数量
        """
        child_number = 0
        can_move = maze.can_move_actions(node.loc)
        
        for action in can_move:
            new_loc = tuple(node.loc[i] + move_map[action][i] for i in range(2))
            if not is_visit_m[new_loc]:
                child = SearchTree(loc=new_loc, action=action, parent=node)
                node.add_child(child)
                child_number += 1
        return child_number

    def back_propagation(node):
        """
        从目标节点回溯到起始节点，构建完整路径
        Args:
            node: 目标节点
        Returns:
            list: 从起点到终点的动作序列
        """
        path = []
        while node.parent is not None:
            path.insert(0, node.to_this_action)
            node = node.parent
        return path

    def myDFS(maze):
        """
        实现深度优先搜索算法
        Args:
            maze: 迷宫对象
        Returns:
            list: 从起点到终点的路径
        """
        start = maze.sense_robot()
        root = SearchTree(loc=start)
        stack = [root]  # 使用列表模拟栈
        h, w, _ = maze.maze_data.shape
        is_visit_m = np.zeros((h, w), dtype=np.int)
        
        while stack:
            current_node = stack[-1]  # 获取栈顶节点

            # 检查是否到达目标
            if current_node.loc == maze.destination:
                return back_propagation(current_node)

            # 如果当前节点是叶子节点且未访问过
            if current_node.is_leaf() and not is_visit_m[current_node.loc]:
                is_visit_m[current_node.loc] = 1
                child_number = expand(maze, is_visit_m, current_node)
                
                if child_number == 0:  # 如果没有子节点，回溯
                    stack.pop()
                else:  # 将新生成的子节点加入栈
                    stack.extend(current_node.children)
            else:
                # 如果当前节点已访问或无路可走，回溯
                stack.pop()

        return []  # 如果没有找到路径，返回空列表

    return myDFS(maze)

class Robot(TorchRobot):
    """
    基于DQN的迷宫求解机器人
    继承自TorchRobot类，实现了深度Q学习算法
    """
    def __init__(self, maze):
        """
        初始化机器人
        Args:
            maze: 迷宫对象
        """
        super(Robot, self).__init__(maze)
        # 设置奖励函数
        maze.set_reward(reward={
            "hit_wall": 5.0,  # 撞墙惩罚
            "destination": -maze.maze_size ** 2.0,  # 到达目标的奖励
            "default": 1.0,  # 默认移动奖励
        })
        self.maze = maze
        self.epsilon = 0  # 探索率
        self.memory.build_full_view(maze=maze)  # 构建完整的环境视图
        self.loss_list = self.train()  # 开始训练

    def train(self):
        """
        训练DQN模型
        Returns:
            list: 训练过程中的损失值列表
        """
        loss_list = []
        batch_size = len(self.memory)

        while True:
            loss = self._learn(batch=batch_size)  # 执行一步学习
            loss_list.append(loss)
            self.reset()  # 重置环境
            
            # 测试当前策略
            for _ in range(self.maze.maze_size ** 2 - 1):
                a, r = self.test_update()
                if r == self.maze.reward["destination"]:
                    return loss_list

    def train_update(self):
        """
        执行一步训练更新
        Returns:
            tuple: (动作, 奖励)
        """
        def state_train():
            return self.sense_state()

        def action_train(state):
            return self._choose_action(state)

        def reward_train(action):
            return self.maze.move_robot(action)

        state = state_train()
        action = action_train(state)
        reward = reward_train(action)
        return action, reward

    def test_update(self):
        """
        执行一步测试更新
        Returns:
            tuple: (动作, 奖励)
        """
        def state_test():
            state = torch.from_numpy(np.array(self.sense_state(), dtype=np.int16)).float().to(self.device)
            return state

        state = state_test()
        self.eval_model.eval()
        with torch.no_grad():
            q_value = self.eval_model(state).cpu().data.numpy()

        def action_test(q_value):
            return self.valid_action[np.argmin(q_value).item()]

        def reward_test(action):
            return self.maze.move_robot(action)

        action = action_test(q_value)
        reward = reward_test(action)
        return action, reward

