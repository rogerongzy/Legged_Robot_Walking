import numpy as np
from robot import Bipedal
import tform as tf
import time
from walking_generator import PreviewControl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    bipedal = Bipedal()
    # zc = 0.45  # Center of Mass height [m]
    stride_x = 0.15 # 步长，0.08可以较好的平稳度过起伏，0.15
    stride_y = 0.18 # 0.18
    # CoM_to_body = np.array([0.0, 0.0, 0.0])  # from CoM to body coordinate

    targetRPY = [0.0, 0.0, -0.02] # 改变yaw，机器人面向正前方走斜线，斜线角度=yaw，取正为偏右，-0.02恰好直线向前
    # plane-yaw:-0.02; 
    pre = PreviewControl(Tsup_time=0.3, Tdl_time=0.2, previewStepNum=180)  # preview control
    bipedal.positionInitialize(initializeTime=0.2)
    CoMTrajectory = np.empty((0, 3), float)
    RobotHeightTrj = np.empty((0, 1), float)

    trjR_log = np.empty((0, 3), float)
    trjL_log = np.empty((0, 3), float)
    walkingCycle = 60 # 步数，plane:30; step:60
    theta = 0.0  # "-" turn left, "+" turn right, is it opposite?(let x = y, it seems become right?)

    supPoint = np.array([0., stride_y / 2])  # ZMP估计位置，两脚中心到脚底板外侧距离为0.1239，两脚中心到电机中心距离0.058，考虑到行走时两脚间距，这个值取大一些，取正代表左脚支撑
    for w in range(walkingCycle):
        # generate one cycle trajectory
        # print(bipedal.getRobotPosition())
        t1 = time.time()
        comTrj, footTrjL, footTrjR = pre.footPrintAndCOMtrajectoryGenerator(inputTargetZMP=supPoint,
                                                                            inputFootPrint=supPoint, Comheight=0.4)
        dt = time.time() - t1
        # print(dt) # 采样时间间隔
        # if supPoint[0] == 0.15:
        #     print("")
        CoMTrajectory = np.vstack((CoMTrajectory, comTrj))
        trjR_log = np.vstack((trjR_log, footTrjR))
        trjL_log = np.vstack((trjL_log, footTrjL))

        com_len = len(comTrj)
        for i in range(com_len):
            targetPositionR = footTrjR[i] - comTrj[i]
            targetPositionL = footTrjL[i] - comTrj[i]
            # print(bipedal.R)

            PosR = bipedal.inverseKinematics(targetPositionR, targetRPY, bipedal.R)
            # PosR = bipedal.inverseKinematics2(targetPositionR - [0, -0.075, -0.13], tf.getRotationPitch(0.0))
            # PosL = bipedal.inverseKinematics2(targetPositionL - [0, 0.075, -0.13], tf.getRotationPitch(0.0))
            PosL = bipedal.inverseKinematics(targetPositionL, targetRPY, bipedal.L)
            # print("PosL=", PosL[2:5])
            # print(bipedal.getEuler())
            bipedal.jointcontroller(PosR, targetRPY)
            bipedal.jointcontroller(PosL, targetRPY)
            bipedal.setLeftLegJointPositions(PosL)
            bipedal.setRightLegJointPositions(PosR)

            # print(bipedal.getRobotPosition()[2])
            RobotHeightTrj = np.vstack((RobotHeightTrj, bipedal.getRobotPosition()[2]))

            bipedal.oneStep()

        # supPoint[0] += stride_x
        # supPoint[1] += (-1)**(w + 1) * stride_y
        rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        stride = np.array([stride_x, -(-1) ** w * stride_y])
        ds = np.dot(stride, rot)
        supPoint = supPoint + ds
        # print(supPoint)


    # temporary data saving, 数据个数=(Tsup_time+Tdl_time)*walkingCycle/dt
    np.savetxt('Comtrj.txt', CoMTrajectory, "%f")
    np.savetxt('Robotrj.txt', RobotHeightTrj, "%f")
    np.savetxt('trjL.txt', trjL_log, "%f")
    np.savetxt('trjR.txt', trjR_log, "%f")
    # debug Com
    

    # figure 1
    figx = plt.figure(figsize=(8, 8))
    comx = figx.add_subplot(111)
    comx.plot(CoMTrajectory[:, 0], label="CoMtrj", color="blue") # 蓝线呈现平滑过渡到直线，在3D图中反映为先拥挤后均匀，仿真结果为速度从0起步，起步后以恒定stride匀速前进
    comx.plot(pre.px_ref_log[:], label="targetZMP", color="black")
    comx.plot(pre.px, label="ZMP", color="red")
    plt.legend() # 增加legend才会显示上述label内容的图例

    # figure 2
    figy = plt.figure(figsize=(8, 8))
    comy = figy.add_subplot(111)
    comy.plot(CoMTrajectory[:, 1], label="CoMtrj", color="blue") # 横移y方向有规律的类似正弦的摆动，与步态周期的分布类似，起步时由于初始化到ready，会有瞬时的扰动
    comy.plot(pre.py_ref_log[:], label="targetZMP", color="black")
    comy.plot(pre.py, label="ZMP", color="red")
    plt.legend()

    # figure 3
    figz = plt.figure(figsize=(8, 8))
    comz = figz.add_subplot(111)
    comz.plot(RobotHeightTrj[:], label="Robtrj", color="blue") # CoM高度，随采样个数变化，支撑脚替换，小波动属于正常现象
    plt.ylim(bottom = 0, top = 0.8)
    plt.legend()

    # foot trajectory

    # figure 4
    fig = plt.figure(figsize=(8, 8))
    ax1 = Axes3D(fig)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.plot(trjL_log[:, 0], trjL_log[:, 1], trjL_log[:, 2], marker="o", linestyle='None') # left leg ending position
    ax1.plot(trjR_log[:, 0], trjR_log[:, 1], trjR_log[:, 2], marker="o", linestyle='None') # right leg ending position
    ax1.plot(CoMTrajectory[:, 0], CoMTrajectory[:, 1], CoMTrajectory[:, 2], marker="o", linestyle='None') # trajectory of CoM (3D Version)
    plt.show()

    bipedal.disconnect()


# CoMTrajectory: CoM轨迹的三维数据（仿真环境下），[x-前进方向坐标,y-横移方向坐标,z-垂直高度坐标]