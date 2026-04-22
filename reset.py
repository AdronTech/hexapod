import Adafruit_PCA9685
import time

frequency = 50

def move_servo(controller, id, value):
    controller.set_pwm(id, 0, int(frequency * 4096 * value / 1000.0))

left = Adafruit_PCA9685.PCA9685(address=0x40)
right = Adafruit_PCA9685.PCA9685(address=0x41)

left.set_pwm_freq(frequency)
right.set_pwm_freq(frequency)

value = 0

move_servo(right, 0, value)
move_servo(right, 1, value)
move_servo(right, 3, value)

move_servo(right, 4, value)
move_servo(right, 5, value)
move_servo(right, 6, value)

move_servo(right, 8, value)
move_servo(right, 9, value)
move_servo(right, 10, value)

move_servo(left, 15, value)
move_servo(left, 14, value)
move_servo(left, 13, value)

move_servo(left, 11, value)
move_servo(left, 10, value)
move_servo(left, 9, value)

move_servo(left, 7, value)
move_servo(left, 6, value)
move_servo(left, 5, value)