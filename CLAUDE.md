# HEXAPOD

This project is about a hexapod robot that uses the waveshare st3020 servos.
They are connected via a waveshare bus servo adapter board which has an USB to TTL converter and handles the half-duplex line control automatically.
The protocol of the servos is heavily inspired by the Dynamixel protocol v1.

This project is the control software that runs on a connected Laptop or raspberry pi and is written in Python. The dependency manager is uv.

The Servo IDs are structured as follows:

Servo ID: <LEG_Index> * 10 + <JOINT_Index>
JOINT_INDEX: 1: Coxa, 2: Femur, 3: Tibia

The LEG_INDEX is 1=Front-Right, 2=Middle-Right, 3=Rear-Right, 4=Rear-Left, 5=Middle-Left, 6=Front-Left

  FRONT
  6  1
5      2
  4  3

More information:
Kinematics: @docs/kinematics.md