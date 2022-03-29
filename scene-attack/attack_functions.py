import numpy as np


class SmoothTurn:
    def __init__(self, attack_power, pow, border):
        """
        Initializes a SmoothTurn transformation which makes a simple turn in the road
        """
        self.b = pow
        self.a = attack_power / (self.b * 10 ** self.b)
        self.border = border

    def f(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = np.zeros_like(x)
        ret[x < 10] = self.a * x[x < 10] ** self.b
        ret[x >= 10] = 10 ** (self.b - 1) * self.a * self.b * x[x >= 10] - (self.b - 1) * self.a * 10 ** self.b
        realret = np.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret

    def f_prime(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = np.zeros_like(x)
        ret[x < 10] = self.a * self.b * x[x < 10] ** (self.b - 1)
        ret[x >= 10] = (self.b - 1) * self.a * self.b
        realret = np.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret

    def f_zegond(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = np.zeros_like(x)
        ret[x < 10] = self.a * self.b * (self.b - 1) * x[x < 10] ** (self.b - 2)
        realret = np.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret


class DoubleTurn:
    def __init__(self, attack_power, pow, d, border):
        """
        Initializes a Double Turn transformation which is to consecutive SmoothTurns in opposite directions
        """
        self.smooth_func = SmoothTurn(attack_power, pow, border)
        self.d = d

    def f(self, x):
        return self.smooth_func.f(x) - self.smooth_func.f(x - self.d)

    def f_prime(self, x):
        return self.smooth_func.f_prime(x) - self.smooth_func.f_prime(x - self.d)

    def f_zegond(self, x):
        return self.smooth_func.f_zegond(x) - self.smooth_func.f_zegond(x - self.d)


class RippleRoad:
    def __init__(self, attack_power, l, border):
        """
        Initializes a RippleRoad transformation which makes a ripple road
        """
        self.attack_power = attack_power
        self.l = l
        self.border = border

    def f(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = self.attack_power * (1 - np.cos(2 * np.pi * x / self.l))
        realret = np.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret

    def f_prime(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = self.attack_power * 2 * np.pi / self.l * np.sin(2 * np.pi * x / self.l)
        realret = np.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret

    def f_zegond(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = self.attack_power * 4 * np.pi**2 / self.l**2 * np.cos(2 * np.pi * x / self.l)
        realret = np.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret


class Combination:
    def __init__(self, params):
        """
        Combines the three types of the above transformations together
        :param params: a dictionary containing parameters for each transformation
        """
        self.smooth_turn = SmoothTurn(params["smooth-turn"]["attack_power"], params["smooth-turn"]["pow"], params["smooth-turn"]["border"])
        self.double_turn = DoubleTurn(params["double-turn"]["attack_power"], params["double-turn"]["pow"], params["double-turn"]["l"], params["double-turn"]["border"])
        self.ripple_road = RippleRoad(params["ripple-road"]["attack_power"], params["ripple-road"]["l"], params["ripple-road"]["border"])

    def f(self, x):
        return self.smooth_turn.f(x) + self.double_turn.f(x) + self.ripple_road.f(x)

    def f_prime(self, x):
        return self.smooth_turn.f_prime(x) + self.double_turn.f_prime(x) + self.ripple_road.f_prime(x)

    def f_zegond(self, x):
        return self.smooth_turn.f_zegond(x) + self.double_turn.f_zegond(x) + self.ripple_road.f_zegond(x)
