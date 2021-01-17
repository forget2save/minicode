# coding:utf-8
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# 这里是全局变量
# 感染率分别为无法感染，低风险，中风险，高风险
InfectiousRate = [0, 0.1, 0.2, 0.3]
# 经过多少时间转换状态，指E->I->R环节，S->E需要感染这个行为触发
DurationThreshold = [0, 50, 50, 1e4]
# 传播半径
Radius = 0.1

# 戴口罩：正方形
# 不戴口罩：圆形
# 传染病四个阶段：Suspective -> Exposed -> Infective -> Recovered
# 对于covid19而言，可以假设所有人都是易感者。
# covid19全局参数
# 基本参数1：低/中/高风险传播率
# S代表易感者
# 基本参数1：是否注意社交距离
# E代表潜伏期病人
# 基本参数1：潜伏期长度
# 基本参数2：潜伏期是否具备传播能力
# I代表出现症状的病人
# 基本参数1：传播范围
# R代表被政府发现已经隔离的人
# 基本参数1：多少天隔离，检测能力

# Person的基本参数
# state：0->1->2->3 状态：S->E->I->R
# pos: (x,y) 显示坐标，主要用于可视化
# roughPos: (i,j) 逻辑坐标，用于逻辑判断
# delta: (v_x, v_y) 相当于离散化的速度
# momentum: 惯性常量
# duration: 记录每个阶段的经过时间，用于状态变更
# masked: 是否佩戴口罩


class Person:
    boundary = np.array([-1, 1, -1, 1], dtype=np.float)

    # 初始化，逻辑坐标从(-1,-1)为原点
    def __init__(self, masked, state=0):
        self.pos = np.random.uniform(low=-1, size=2)
        self.roughPos = ((self.pos - np.array([-1, -1])) / (Radius / 2)).astype(np.int)
        self.delta = np.zeros(2)
        self.masked = masked
        self.state = state
        self.step = 0.02
        self.momentum = 0.95
        self.duration = 0

    # 根据暴露的危险概率患病
    def getVirus(self, dangerLevel):
        if self.state == 0:
            if self.masked:
                dangerLevel -= 1
            if random.random() < InfectiousRate[dangerLevel]:
                self.state = 1

    # 时间流逝，促进某些过程转化
    def timePassBy(self):
        if self.state > 0 and self.state < 3:
            self.duration += random.randint(0, 2)
            if self.duration >= DurationThreshold[self.state]:
                self.state += 1
                self.duration = 0

    # 限制人群移动
    def setBoundary(self, level):
        for i in range(4):
            if i % 2 == 0:
                self.boundary[i] = self.pos[i // 2] - 1 / level
            else:
                self.boundary[i] = self.pos[i // 2] + 1 / level
        self.boundary = np.clip(self.boundary, -1, 1)
        self.step /= math.sqrt(level)

    # 随机移动并更新逻辑坐标
    def randomMovement(self):
        self.delta = self.momentum * self.delta + self.step * np.random.uniform(
            low=-1, size=2
        )
        self.pos += self.delta
        for i in range(2):
            if self.pos[i] < self.boundary[2 * i]:
                self.pos[i] = 2 * self.boundary[2 * i] - self.pos[i]
                self.delta[i] = -self.delta[i]
            elif self.pos[i] > self.boundary[2 * i + 1]:
                self.pos[i] = 2 * self.boundary[2 * i + 1] - self.pos[i]
                self.delta[i] = -self.delta[i]
        self.roughPos = ((self.pos - np.array([-1, -1])) / (Radius / 2)).astype(np.int)


class Crowd:
    outOfBound = 100
    grid_num = int(4 / Radius)
    remove = outOfBound * np.ones(2)
    axis = [-1, 1, -1, 1]
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    shapes = ["o", "s"]
    endFlag = False
    endDelay = 30
    time = [0]
    tmp = [0 for _ in range(4)]

    def __init__(self, masked, unmasked, variety=8, socialDist=1):
        # 新建图层
        self.fig, self.ax = plt.subplots(1, 2, figsize=(9, 4))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        # 基本参数
        self.variety = variety
        self.num = masked + unmasked
        self.masked = masked
        self.unmasked = unmasked
        # 实际坐标*图案总数，outofbound代表不画
        self.pos = [
            self.outOfBound * np.ones((2, self.num)) for _ in range(self.variety)
        ]
        # 以给定人群比例构建对象
        self.people = [Person(i < self.masked) for i in range(self.num)]
        self.people[-1].state = 1
        self.data = [[self.num - 1], [1], [0], [0]]
        # 逻辑坐标Map，并初始化
        self.pos_grid = [[] for i in range(self.grid_num)]
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                self.pos_grid[i].append(set())
        for i in range(self.num):
            x, y = self.people[i].roughPos
            self.pos_grid[x][y].add(i)
        # 构建variety个图层，以不同图案与颜色区分
        plt.subplot(121)
        self.plots = [
            plt.plot(
                self.pos[i][0], self.pos[i][1], self.colors[i // 2] + self.shapes[i % 2]
            )[0]
            for i in range(self.variety)
        ]
        plt.axis(self.axis)
        plt.subplot(122)
        self.lineplots = [
            plt.plot(self.time, self.data[i], self.colors[i])[0] for i in range(4)
        ]
        plt.ylim(0, self.num)
        # 动画展示
        if socialDist > 1:
            [p.setBoundary(socialDist) for p in self.people]
        self.ani = FuncAnimation(
            self.fig, self.animate, frames=range(1, 500), interval=20, repeat=False
        )

    def covid(self):
        for p in self.people:
            if p.state == 1 or p.state == 2:
                x, y = p.roughPos
                resone = [[], [], []]
                resone[0] = [i for i in self.pos_grid[x][y]]
                if y + 1 < self.grid_num:
                    resone[1].extend([i for i in self.pos_grid[x][y + 1]])
                if x + 1 < self.grid_num:
                    resone[1].extend([i for i in self.pos_grid[x + 1][y]])
                if x > 0:
                    resone[1].extend([i for i in self.pos_grid[x - 1][y]])
                if y > 0:
                    resone[1].extend([i for i in self.pos_grid[x][y - 1]])
                if not p.masked:
                    if y + 1 < self.grid_num and x + 1 < self.grid_num:
                        resone[2].extend([i for i in self.pos_grid[x + 1][y + 1]])
                    if y + 1 < self.grid_num and x > 0:
                        resone[2].extend([i for i in self.pos_grid[x - 1][y + 1]])
                    if y > 0 and x > 0:
                        resone[2].extend([i for i in self.pos_grid[x - 1][y - 1]])
                    if y > 0 and x + 1 < self.grid_num:
                        resone[2].extend([i for i in self.pos_grid[x + 1][y - 1]])
                    if y + 2 < self.grid_num:
                        resone[2].extend([i for i in self.pos_grid[x][y + 2]])
                    if x + 2 < self.grid_num:
                        resone[2].extend([i for i in self.pos_grid[x + 2][y]])
                    if x > 1:
                        resone[2].extend([i for i in self.pos_grid[x - 2][y]])
                    if y > 1:
                        resone[2].extend([i for i in self.pos_grid[x][y - 2]])
                helpness = 0
                if p.masked:
                    helpness = 1
                [
                    [self.people[i].getVirus(3 - j - helpness) for i in resone[j]]
                    for j in range(3)
                ]

    def update(self):
        # 人群随机移动并改变状态
        for i, p in enumerate(self.people):
            x, y = p.roughPos
            self.pos_grid[x][y].remove(i)
            p.randomMovement()
            x, y = p.roughPos
            self.pos_grid[x][y].add(i)
            p.timePassBy()
        # 疾病传播
        self.covid()
        self.endFlag = True
        self.tmp = [0 for _ in range(4)]
        # 可视化数据计算
        for i, p in enumerate(self.people):
            for j in range(self.variety):
                self.pos[j][:, i] = self.remove
            self.tmp[p.state] += 1
            tmp = p.state * 2
            if p.masked:
                tmp += 1
            if tmp > 1 and tmp < 6:
                self.endFlag = False
            self.pos[tmp][:, i] = p.pos
        [self.data[i].append(self.tmp[i]) for i in range(4)]

    def animate(self, frame):
        # 数据计算
        if self.endFlag:
            if self.close():
                return
        self.time.append(frame + 1)
        self.update()
        # 可视化
        for i in range(self.variety):
            self.plots[i].set_data(self.pos[i])
        [self.lineplots[i].set_data(self.time, self.data[i]) for i in range(4)]
        plt.xlim(0, self.time[-1])
        return [*self.plots, *self.lineplots]

    def show(self):
        plt.show()

    def close(self):
        if self.endDelay == 0:
            plt.close()
            return True
        self.endDelay -= 1
        return False

    def save(self, path):
        self.ani.save(path, writer=PillowWriter(fps=25))


if __name__ == "__main__":
    # C = Crowd(masked=100, unmasked=0)
    # C.save("covid_all_mask.gif")
    C = Crowd(masked=50, unmasked=50, socialDist=4)
    C.save("covid_social_distance.gif")
    # C = Crowd(masked=0, unmasked=100)
    # C.save("covid_no_mask.gif")
