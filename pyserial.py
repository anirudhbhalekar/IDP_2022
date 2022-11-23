import time
import serial


# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial("COM9", 9600)

ser.close()
ser.open()

timeout = ser.timeout
ser.timeout = 2

writeMode = 1

print('Enter your commands below.\r\nInsert "exit" to leave the application.')
data_in = 1
while 1:

    if writeMode == 0: 
        # get keyboard input
        data_in = input(">> ")
        if data_in == 'exit':
            ser.close()
            exit()
        elif data_in == "WRITE": 
            serial_data = bytes(str(data_in), encoding='utf8')
            ser.write(serial_data)
            writeMode = 1

        else:
            serial_data = bytes(str(data_in), encoding='utf8')
            ser.write(serial_data)
            
    else: 
        ser.write(b"4")
        print(ser.read().decode("utf-8"))
        time.sleep(5)
