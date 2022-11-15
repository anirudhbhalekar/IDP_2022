import time
import serial


# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port='COM9',
    baudrate=9600,
    parity=serial.PARITY_ODD,
    stopbits=serial.STOPBITS_TWO,
    bytesize=serial.SEVENBITS
)

ser.isOpen()
ser.close()
ser.open()

timeout = ser.timeout
ser.timeout = 2


print('Enter your commands below.\r\nInsert "exit" to leave the application.')

data_in = 1
while 1 :
    # get keyboard input
    data_in = input(">> ")
    if data_in == 'exit':
        ser.close()
        ser.timeout = timeout
        exit()
    else:
        _ = ser.read_all()
        # send the character to the device
        # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
        serial_data = bytes(str(data_in), encoding='utf8')
        ser.write(serial_data)
        out = ser.read_until(  )
        # let's wait one second before reading output (let's give device time to answer)
        #time.sleep(1)
        #while ser.inWaiting() > 0:
        #    out += ser.read(1)
            
        #if out != '':
        #    print(">>" + out)