import numpy as np
import tform as tf
import scipy.linalg as la
import control
import swing_trajectory as st
from cvxopt import matrix, solvers


class PreviewControl:
    def __init__(self, dt=1. / 240., Tsup_time=0.5, Tdl_time=0.1, CoMheight=0.35, g=9.8, previewStepNum=240,
                 initialTargetZMP=np.array([0., 0.])):
        self.R = np.array([1.])
        self.Q = np.array([[7000, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.previewStepNum = previewStepNum
        self.dt = dt
        self.CoMheight = 0.4  # 可变参数
        self.A = np.array([[1, self.dt, (self.dt ** 2) / 2],
                           [0, 1, self.dt],
                           [0, 0, 1]])
        self.B = np.mat([(self.dt ** 3) / 6, (self.dt ** 2) / 2, self.dt]).T
        self.C = np.array([1, 0, -self.CoMheight / g])

        # IS-MPC
        # yita = np.sqrt(g / self.CoMheight)
        # self.A = np.array([[np.cosh(yita * self.dt), np.sinh(yita * self.dt) / yita, 1 - np.cosh(yita * self.dt)],
        #                    [yita * np.sinh(yita * self.dt), np.cosh(yita * self.dt), -yita * np.sinh(yita * self.dt)],
        #                    [0, 0, 1]])
        # self.B = np.mat([self.dt - np.sinh(yita * self.dt) / yita, 1 - np.cosh(yita * self.dt), self.dt]).T
        # self.C = np.array([0, 0, 1])
        # self.x = np.mat([0, 0, inputTargetZMP[0]]).T
        # self.y = np.mat([0, 0, inputTargetZMP[1]]).T

        self.G = np.vstack((-self.C * self.B, self.B))
        self.Gr = np.array([[1], [0], [0], [0]])

        self.A_ = np.hstack((np.array([[1], [0], [0], [0]]), np.vstack((np.dot(-self.C, self.A), self.A))))
        P, _, _ = control.dare(self.A_, self.G, self.Q, self.R)  # 求解离散Riccati方程
        tmp = self.A_ - self.G * la.inv(self.R + self.G.T * P * self.G) * self.G.T * P * self.A_
        self.Fr = np.array([])  # 前馈控制的系数，详见梶田秀司的书P141
        for j in range(1, self.previewStepNum + 1):
            self.Fr = np.append(self.Fr,
                                -la.inv(self.R + self.G.T * P * self.G) * self.G.T * (tmp.T ** (j - 1)) * P * self.Gr)

        self.F = -la.inv(self.R + self.G.T * P * self.G) * self.G.T * P * self.A_  # 最优控制的系数 u* = Fx

        self._RIGHT_LEG = 1
        self._LEFT_LEG = 0

        # state vector
        self.x = np.mat([0, 0, 0]).T
        self.y = np.mat([0, 0, 0]).T  # IS-MPC需要修改状态初始值

        # xy坐标为相对世界坐标系的绝对坐标
        self.footPrints = np.array([[[0., 0.075], [0., -0.075]],  # 摆动脚起始位置(右、左)[不管那只脚先迈，始终都是右左的排步]
                                    [[0., 0.075], [0., -0.075]],  # 摆动脚终止位置
                                    [[0., 0.075], [0., -0.075]]])  # 支撑脚位置 3*2*2,执行delete操作时，3当作行看待，第一个2当作列看待

        self.Tsup = int(Tsup_time / dt)
        self.Tdl = int(Tdl_time / dt)

        self.px_ref = np.full((self.Tsup + self.Tdl) * 3, initialTargetZMP[0])  # 维度>previewStepNum+len(input_px_ref)
        self.py_ref = np.full((self.Tsup + self.Tdl) * 3, initialTargetZMP[1])
        self.px = np.array([0.0])  # zmp
        self.py = np.array([0.0])

        self.px_ref_log = self.px_ref[:(self.Tsup + self.Tdl) * 2]  # 维度大小无所谓
        self.py_ref_log = self.py_ref[:(self.Tsup + self.Tdl) * 2]

        self.xdu = 0
        self.ydu = 0

        self.xu = 0
        self.yu = 0

        self.dx = np.mat(np.zeros(3)).T
        self.dy = np.mat(np.zeros(3)).T

        self.swingLeg = self._RIGHT_LEG
        self.supportLeg = self._LEFT_LEG

        self.targetZMPold = np.array([initialTargetZMP])

        self.currentFootStep = 0

    def footPrintAndCOMtrajectoryGenerator(self, inputTargetZMP, inputFootPrint, Comheight, g=9.8):
        # self.CoMheight = Comheight  # 可变参数
        # self.A = np.array([[1, self.dt, (self.dt ** 2) / 2],
        #                    [0, 1, self.dt],
        #                    [0, 0, 1]])
        # self.B = np.mat([(self.dt ** 3) / 6, (self.dt ** 2) / 2, self.dt]).T
        # self.C = np.array([1, 0, -self.CoMheight / g])
        #
        # # IS-MPC
        # # yita = np.sqrt(g / self.CoMheight)
        # # self.A = np.array([[np.cosh(yita * self.dt), np.sinh(yita * self.dt) / yita, 1 - np.cosh(yita * self.dt)],
        # #                    [yita * np.sinh(yita * self.dt), np.cosh(yita * self.dt), -yita * np.sinh(yita * self.dt)],
        # #                    [0, 0, 1]])
        # # self.B = np.mat([self.dt - np.sinh(yita * self.dt) / yita, 1 - np.cosh(yita * self.dt), self.dt]).T
        # # self.C = np.array([0, 0, 1])
        # # self.x = np.mat([0, 0, inputTargetZMP[0]]).T
        # # self.y = np.mat([0, 0, inputTargetZMP[1]]).T
        #
        # self.G = np.vstack((-self.C * self.B, self.B))
        # self.Gr = np.array([[1], [0], [0], [0]])
        #
        # self.A_ = np.hstack((np.array([[1], [0], [0], [0]]), np.vstack((np.dot(-self.C, self.A), self.A))))
        # P, _, _ = control.dare(self.A_, self.G, self.Q, self.R)  # 求解离散Riccati方程
        # tmp = self.A_ - self.G * la.inv(self.R + self.G.T * P * self.G) * self.G.T * P * self.A_
        # self.Fr = np.array([])  # 前馈控制的系数，详见梶田秀司的书P141
        # for j in range(1, self.previewStepNum + 1):
        #     self.Fr = np.append(self.Fr,
        #                         -la.inv(self.R + self.G.T * P * self.G) * self.G.T * (tmp.T ** (j - 1)) * P * self.Gr)
        #
        # self.F = -la.inv(self.R + self.G.T * P * self.G) * self.G.T * P * self.A_  # 最优控制的系数 u* = Fx

        currentFootStep = 0

        self.footPrints = self.footOneStep(self.footPrints, inputFootPrint, self.supportLeg)
        # print(self.footPrints)

        input_px_ref, input_py_ref = self.targetZMPgenerator(inputTargetZMP, self.targetZMPold[-1], self.Tsup, self.Tdl)

        self.px_ref = self.fifo(self.px_ref, input_px_ref, len(input_px_ref))
        self.py_ref = self.fifo(self.py_ref, input_py_ref, len(input_py_ref))

        self.px_ref_log = np.append(self.px_ref_log, input_px_ref)
        self.py_ref_log = np.append(self.py_ref_log, input_py_ref)

        CoMTrajectory = np.empty((0, 3), float)
        # RobotstartVelocity = np.array([self.x[1], self.y[1], 0.0])
        for k in range(len(input_px_ref)):
            dpx_ref = self.px_ref[k + 1] - self.px_ref[k]
            dpy_ref = self.py_ref[k + 1] - self.py_ref[k]

            xe = self.px_ref[k] - self.C * self.x
            ye = self.py_ref[k] - self.C * self.y

            X = self.A_ * np.vstack((xe, self.dx)) + self.G * self.xdu + self.Gr * dpx_ref
            Y = self.A_ * np.vstack((ye, self.dy)) + self.G * self.ydu + self.Gr * dpy_ref

            xsum = ysum = 0
            for j in range(1, self.previewStepNum + 1):
                xsum += self.Fr[j - 1] * (self.px_ref[k + j] - self.px_ref[k + j - 1])
                ysum += self.Fr[j - 1] * (self.py_ref[k + j] - self.py_ref[k + j - 1])

            self.xdu = self.F * X + xsum
            self.ydu = self.F * Y + ysum

            self.xu += self.xdu
            self.yu += self.ydu

            old_x = self.x
            old_y = self.y

            self.x = self.A * self.x + self.B * self.xu
            self.y = self.A * self.y + self.B * self.yu

            self.dx = self.x - old_x
            self.dy = self.y - old_y

            CoMTrajectory = np.vstack((CoMTrajectory, [self.x[0, 0], self.y[0, 0], self.CoMheight]))

            self.px = np.append(self.px, self.C * self.x)
            self.py = np.append(self.py, self.C * self.y)

        # RobotEndVelocity = np.array([self.x[1], self.y[1], 0.0])

        leftTrj, rightTrj = self.footTrajectoryGenerator(
            np.hstack((self.footPrints[currentFootStep, self.swingLeg], 0.)),  # 维度1*3，填充的0表示z的值
            np.hstack((self.footPrints[currentFootStep + 1, self.swingLeg], 0.)),
            np.array([0.0,0.0,0.0]),  # RobotstartVelocity,
            np.array([0.0,0.0,0.0]),  # RobotEndVelocity,
            np.hstack((self.footPrints[currentFootStep, self.supportLeg], 0.)),
            self.swingLeg)

        self.swingLeg, self.supportLeg = self.changeSupportLeg(self.swingLeg, self.supportLeg)
        self.targetZMPold = np.vstack((self.targetZMPold, inputTargetZMP))
        # print(self.targetZMPold[-1]) 最后一行

        return CoMTrajectory, leftTrj, rightTrj

    def targetZMPgenerator(self, targetZMP, targetZMPold, Tsup, Tdl):
        tdl_t = np.arange(0, Tdl)
        x_a = (targetZMPold[0] - targetZMP[0]) / (0 - Tdl)
        x_b = targetZMPold[0]
        y_a = (targetZMPold[1] - targetZMP[1]) / (0 - Tdl)
        y_b = targetZMPold[1]

        px_ref = np.hstack((x_a * tdl_t + x_b, np.full(Tsup, targetZMP[0])))
        py_ref = np.hstack((y_a * tdl_t + y_b, np.full(Tsup, targetZMP[1])))

        return px_ref, py_ref

    def footTrajectoryGenerator(self, swingStartPointV, swingEndPointV, startRobotVelocityV_xy, endRobotVelocityV,
                                supportPointV, swingLeg, zheight=0.035):
        # zheight : 抬脚高度
        supportTrajectory = np.vstack((np.full(self.Tdl + self.Tsup, supportPointV[0]),
                                       np.full(self.Tdl + self.Tsup, supportPointV[1]),
                                       np.full(self.Tdl + self.Tsup, supportPointV[2]))).T

        swingTrajectoryForTdl = np.vstack((np.full(self.Tdl, swingStartPointV[0]),
                                           np.full(self.Tdl, swingStartPointV[1]),
                                           np.full(self.Tdl, swingStartPointV[2]))).T

        if np.array_equal(swingStartPointV, swingEndPointV):
            swingTrajectoryForTsup = np.vstack((np.full(self.Tsup, swingEndPointV[0]),
                                                np.full(self.Tsup, swingEndPointV[1]),
                                                np.full(self.Tsup, swingEndPointV[2]))).T

        else:
            swingTrajectoryForTsup = st.swingTrajectoryGenerator(swingStartPointV, swingEndPointV,
                                                                 -startRobotVelocityV_xy, -endRobotVelocityV, zheight,
                                                                 0., self.Tsup * self.dt, self.dt)

        if swingLeg is self._RIGHT_LEG:
            trjR = np.vstack((swingTrajectoryForTdl, swingTrajectoryForTsup))
            trjL = supportTrajectory
        elif swingLeg is self._LEFT_LEG:
            trjL = np.vstack((swingTrajectoryForTdl, swingTrajectoryForTsup))
            trjR = supportTrajectory

        return trjL, trjR

    def fifo(self, p, in_p, range, vstack=False):
        if vstack:
            return np.vstack((np.delete(p, range, 0), in_p))

        else:
            return np.append(np.delete(p, slice(range), None), in_p)

    def footOneStep(self, footPrints, supportPoint, supportLeg):
        step = len(footPrints)
        # 下一步的位置逐渐顶替之前的数据
        if supportLeg is self._LEFT_LEG:
            newFootPrint = np.vstack((footPrints, [np.vstack((supportPoint, footPrints[-1, 1]))]))  # -1代表最后一行

        elif supportLeg is self._RIGHT_LEG:
            newFootPrint = np.vstack((footPrints, [np.vstack((footPrints[-1, 0], supportPoint))]))

        return np.delete(newFootPrint, 0, 0)  # 删去第一行，完成数据顶替

    def changeSupportLeg(self, swingLeg, supportLeg):
        return supportLeg, swingLeg
