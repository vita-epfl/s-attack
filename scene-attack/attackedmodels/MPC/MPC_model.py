import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.interpolate import CubicSpline
from casadi import *
import pickle


def get_best_traj_points(history, centerline, speed):
    riding_lane_points = centerline
    new_riding_lane_points = np.zeros((len(riding_lane_points) + 3, 2))
    new_riding_lane_points[3:, :] = riding_lane_points
    new_riding_lane_points[:3, :] = history[-3:, :]
    riding_lane_points = new_riding_lane_points
    # _, idx = np.unique(riding_lane_points[:, 0], return_index=True)
    # riding_lane_points = riding_lane_points[idx, :]
    # riding_lane_points = riding_lane_points[riding_lane_points[:, 0].argsort(), :]
    t = np.arange(len(riding_lane_points))
    csx = CubicSpline(t, riding_lane_points[:, 0])
    csy = CubicSpline(t, riding_lane_points[:, 1])


    d = speed / 10

    points = []
    cur_time = 2
    for step in range(30):
        dx, dy = csx(cur_time, 1), csy(cur_time, 1)
        delta_t = d / np.sqrt(dx**2 + dy**2)
        new_time = cur_time + delta_t

        left, right = cur_time, len(riding_lane_points)
        mx_dist = np.sqrt((csx(right) - csx(cur_time))**2 + (csy(right) - csy(cur_time))**2)
        if mx_dist < d:
            right *= 2
        while True:
            dist = np.sqrt((csx(new_time) - csx(cur_time))**2 + (csy(new_time) - csy(cur_time))**2)
            if np.abs(dist - d) <= 0.01:
                break
            if dist > d:
                right = new_time
            elif dist < d:
                left = new_time
            new_time = (left + right) / 2

        cur_time = new_time
        points.append(np.array([csx(cur_time), csy(cur_time)]))
    points = np.array(points)

    # xs = np.linspace(0, 300, 100)
    # interpolated_lane = np.zeros((100, 2))
    # interpolated_lane[:, 0] = xs
    # interpolated_lane[:, 1] = cs(xs)
    # visualize(np.array([interpolated_lane]))
    # plt.scatter(points[:, 0], points[:, 1])
    # visualize(np.array([interpolated_lane]), True)
    # plt.plot(xs, cs(xs), "-",
    #             color="b",
    #             label="interpolated",
    #             alpha=1,
    #             linewidth=5,
    #             zorder=0,)
    # plt.plot(riding_lane_points[:, 0], riding_lane_points[:, 1], "-",
    #          color="g",
    #          label="found lane",
    #          alpha=1,
    #          linewidth=5,
    #          zorder=0, )
    return points


def mpc_fun(best_mode_prediction_kd, sample_rate, pixel_scale, n_pred, observation):
    # pdb.set_trace()
    obs = observation
    y_ref = best_mode_prediction_kd[:, 1]
    x_ref = best_mode_prediction_kd[:, 0]
    y_ref = np.concatenate((np.array([obs[-1, 1]]), y_ref), axis=0)
    x_ref = np.concatenate((np.array([obs[-1, 0]]), x_ref), axis=0)
    # delta_t = sample_rate
    delta_t = 0.1
    CONVERSION = pixel_scale[0]
    scale = pixel_scale[0]
    N = n_pred + 1
    opti = casadi.Opti()
    x = opti.variable(1, N)
    y = opti.variable(1, N)
    psi = opti.variable(1, N)
    v = opti.variable(1, N)
    vx = opti.variable(1, N)  # ahmad
    vy = opti.variable(1, N)  # ahmad
    beta = opti.variable(1, N)

    u1 = opti.variable(1, N - 1)  # acceleration
    u2 = opti.variable(1, N - 1)  # angle
    acc = opti.variable(2, N - 1)  # ahmad

    err_u1 = opti.variable(1, N - 1)
    err_u2 = opti.variable(1, N - 1)
    for k in range(N - 2):
        opti.subject_to(err_u1[k + 1] == u1[k + 1] - u1[k])
        opti.subject_to(err_u2[k + 1] == u2[k + 1] - u2[k])

    p = opti.parameter(4, 1)
    opti.minimize(
        10 * sumsqr(y - np.array(y_ref, ndmin=2)) + 10 * sumsqr(x - np.array(x_ref, ndmin=2)) + 0.1 * sumsqr(
            err_u1) + 5 * sumsqr(err_u2) + 0.1 * sumsqr(u1))

    opti.subject_to(u1 <= 4 * scale)
    opti.subject_to(u2 <= 45 * np.pi / 180)
    opti.subject_to(u1 >= -4 * scale)
    opti.subject_to(u2 >= -45 * np.pi / 180)

    x[0] = p[0]
    y[0] = p[1]
    psi[0] = p[2]
    v[0] = p[3]
    for k in range(N - 1):
        opti.subject_to(beta[k] == np.arctan(np.tan(u2[k]) * 1.77 / (1.77 + 1.17)))
        # best model
        # opti.subject_to(beta[k] == np.arctan(np.tan(u2[k])/2))
        opti.subject_to(x[k + 1] == x[k] + v[k] * delta_t * cos(psi[k] + beta[k]))
        opti.subject_to(y[k + 1] == y[k] + v[k] * delta_t * sin(psi[k] + beta[k]))

        opti.subject_to(vx[k] * delta_t == x[k+1] - x[k])  # ahmad
        opti.subject_to(vy[k] * delta_t == y[k+1] - y[k])  # ahmad
        opti.subject_to(acc[0, k] * delta_t == vx[k + 1] - vx[k])  # ahmad
        opti.subject_to(acc[1, k] * delta_t == vy[k + 1] - vy[k])  # ahmad
        # best model
        # opti.subject_to(psi[k + 1] == psi[k] + v[k] / (1.5 * CONVERSION) * delta_t * sin(beta[k]))
        opti.subject_to(psi[k + 1] == psi[k] + v[k] / (1.77 * CONVERSION) * delta_t * sin(beta[k]))
        # opti.subject_to(v[k + 1] == v[k] + u1[k] * delta_t)
        opti.subject_to(u1[k]**2 == acc[0, k]**2 + acc[1, k]**2)  # ahmad

    opti.print_header = False
    opti.print_iteration = False
    opti.print_time = False
    opti.solver('ipopt')
    pixel_d = np.sqrt((obs[-1, 1] - obs[-2, 1]) ** 2 + (obs[-1, 0] - obs[-2, 0]) ** 2)
    velocity = pixel_d / delta_t
    psi_ref = np.arctan2((obs[-1, 1] - obs[-2, 1]), (obs[-1, 0] - obs[-2, 0]))

    # pdb.set_trace()
    opti.set_value(p, [obs[-1, 0], obs[-1, 1], psi_ref, velocity])

    try:
        sol = opti.solve()

        x = np.expand_dims(sol.value(x), axis=1)
        y = np.expand_dims(sol.value(y), axis=1)
        # print("vx:", sol.value(vx))
        # print("vy:", sol.value(vy))
        # print("acc x:", sol.value(acc)[0, :])
        # print("acc y:", sol.value(acc)[1, :])
        # print("u1:", sol.value(u1))
        # print("test acc x:", np.abs(sol.value(u1)**2 - sol.value(acc)[0, :]**2 - sol.value(acc)[1, :]**2))
        return np.concatenate((x, y), axis=1)
    except:
        return np.array([[0, 0], [0, 0]])


def load_obj(name ):
    with open("objs/" + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_pred(centerline, history):
    current_speed = np.sqrt(((history[-1] - history[-2]) ** 2).sum()) * 10
    points = get_best_traj_points(history, centerline, current_speed)
    pred = mpc_fun(points, 1, np.array([1]), 30, history)
    return pred[1:]


if __name__ == '__main__':
    for i in range(10):
        # st = load_obj("st_of_scenario_"+str(i))
        # history = get_history(st)
        #
        # lanes = np.load("objs/scenario_"+str(i)+"_lanes.npy")
        # visualize(lanes)
        # current_speed = np.sqrt(((history[-1] - history[-2]) ** 2).sum()) * 10
        # points = get_best_traj_points(lanes, current_speed)
        # # print(points.shape)
        # # speed = np.sqrt(((points[1:] - points[:-1])**2).sum(1))
        # # print(speed.mean(), ((speed - speed.mean())**2).mean())
        # # visualize(np.array([interpolated_lane]))
        # pred = mpc_fun(points, 1, np.array([1]), 30, history)
        d = load_obj("am_lines/am_lines_"+str(i))
        history = d["obs"]
        lanes = np.array([d["centerline"]])
        best_path = d["centerline"]
        t = np.arange(len(best_path))
        csx = CubicSpline(t, best_path[:, 0])
        csy = CubicSpline(t, best_path[:, 1])
        plt.scatter(csx(t), csy(t))
        for time in t:
            dx = csx(time, 1)
            dy = csy(time, 1)
            xs = csx(time) + np.linspace(0, 0.5) * dx
            ys = csy(time) + np.linspace(0, 0.5) * dy
            plt.plot(xs, ys, color="r")
        # plt.xlim(-30, 30)
        # plt.ylim(-30, 30)
        for lane_cl in lanes:
            # lane_cl = np.matmul(st["rot"].T, lane_cl.T).T + st["orig"].reshape(-1, 2)
            plt.plot(
                lane_cl[:, 0],
                lane_cl[:, 1],
                "-",
                color="#5A5A5B",
                alpha=1,
                linewidth=1,
                zorder=0,
            )

        plt.xlabel("Map X")
        plt.ylabel("Map Y")

        color_dict = {"AGENT_HISTORY": "#FCE303", "AGENT_GT": "#03FC0F",
                      "AGENT_PRED0": "#FF0000", "AGENT_PRED1": "#FF7300", "AGENT_PRED2": "#FFA200",
                      "OTHERS": "#274B8B", "AV": "#007672", "AGENT_HISTORY_ORIG": "#804600",
                      "AGENT_PRED_ORIG": "#DE00CB"}
        plt.plot(
            history[:, 0],
            history[:, 1],
            "-",
            color=color_dict["AGENT_HISTORY"],
            label='Agent history',
            alpha=1,
            linewidth=2,
            zorder=0,
        )
        # plt.plot(
        #     pred[:, 0],
        #     pred[:, 1],
        #     "-",
        #     color=color_dict["AGENT_PRED0"],
        #     label='Agent Pred ',
        #     alpha=1,
        #     linewidth=2,
        #     zorder=0,
        # )
        # plt.scatter(pred[-1, 0], pred[-1, 1],
        #             c=color_dict["AGENT_PRED0"], s=50)
        plt.scatter(history[-1, 0], history[-1, 1], marker='X', c=color_dict["AGENT_HISTORY"], s=50,
                    label="Attack start point")
        plt.legend()
        plt.show()


