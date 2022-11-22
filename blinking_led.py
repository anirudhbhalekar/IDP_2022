from datetime import datetime
import serial

# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial("COM9", 9600)

ser.close()
ser.open()

timeout = ser.timeout
ser.timeout = 2

prev_time = datetime.now()
current_state = 0

for i in range(10 ** 7):
    time = datetime.now()
    dif = (time - prev_time).microseconds / 10 ** 6
    if dif > 0.25:
        prev_time = time
        time = datetime.now()
        current_state = 1 - current_state
        command = "32" + str(current_state)
        serial_data = bytes(command, encoding='utf8')

        ser.write(serial_data)
    #serial_data = bytes(str(data_in), encoding='utf8')
    #ser.write(serial_data)

command = "320"
print(command)
serial_data = bytes(command, encoding='utf8')
ser.write(serial_data)