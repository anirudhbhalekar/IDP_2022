import time
import serial


# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial("COM9", 9600)

ser.close()
ser.open()

timeout = ser.timeout
ser.timeout = 2

writeMode = 0 

print('Enter your commands below.\r\nInsert "exit" to leave the application.')
data_in = 1

prev_time = initial
for i in range(10):
    serial_data = bytes(str(data_in), encoding='utf8')
    ser.write(serial_data)
