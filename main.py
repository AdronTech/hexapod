import Adafruit_PCA9685
import time
from math import acos, degrees, sqrt, asin

frequency = 200

tibia = 100
femur = 73
coxa = 36

left = Adafruit_PCA9685.PCA9685(address=0x40)
right = Adafruit_PCA9685.PCA9685(address=0x41)

left.set_pwm_freq(frequency)
right.set_pwm_freq(frequency)


def move_servo(controller, id, value):
    controller.set_pwm(id, 0, int(frequency * 4096 * value / 1000.0))


def move_joint_leg(leg_nr, joint_nr, value):

    is_left = leg_nr >= 3

    # map value to pulse-length
    if is_left:
        value *= -1

    pulse = value / 45.0 * 0.5 + 1.5

    # get right controller
    if is_left:
        controller = left
    else:
        controller = right

    # get right id
    if is_left:
        leg_nr -= 3

    id = leg_nr * 4 + joint_nr

    if is_left:
        id = 15 - id

    # move servo
    move_servo(controller, id, pulse)


def move_leg(leg_nr, x, y, z):

    l_2 = sqrt(y**2 + x**2)

    gamma = degrees(asin(y/l_2))

    l_1 = sqrt(z**2 + l_2**2)
    beta_2 = degrees(asin(x / l_1))

    alpha = degrees(acos((femur**2 + tibia**2 - l_1**2)/(2.0*femur*tibia)))
    beta = degrees(acos((femur**2 + l_1**2 - tibia**2)/(2.0*femur*l_1)))

    alpha = 90 - alpha -22
    beta = 90 - beta - beta_2

    # print(alpha)
    # print(beta)

    move_joint_leg(leg_nr, 0, alpha)
    move_joint_leg(leg_nr, 1, beta)
    move_joint_leg(leg_nr, 2, gamma)

def move_group(value):
    offset_x = 65
    move_leg(0, offset_x, value, 100)
    move_leg(2, offset_x, value, 100)
    move_leg(4, offset_x, value, 100)
    
    move_leg(1, offset_x, value, 100)
    move_leg(3, offset_x, value, 100)
    move_leg(5, offset_x, value, 100)

def interpolate(minimum, maximum, norm):
    return (maximum - minimum) * norm + minimum


timestep = 0.01
duration = 0.5

minimum= -30
maximum = 30

while True:

    t = 0

    while t < duration:

        value = interpolate(minimum, maximum, t/duration)
        move_group(value)

        t += timestep
        time.sleep(timestep)

    t = 0

    while t < duration:

        value = interpolate(maximum, minimum, t/duration)
        move_group(value)

        t += timestep
        time.sleep(timestep)


# move_joint_leg(0,0,0)
# move_joint_leg(0,1,0)

# for i in range(6):
#     for j in range(4):
#         move_joint_leg(i, j, 0)

#         time.sleep(0.2)
