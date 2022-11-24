import time
import serial


# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial("COM9", 9600)

ser.close()
ser.open()

timeout = ser.timeout
ser.timeout = 2
dec_val_list = []

writeMode = 0

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
        raw_read = ser.read(2)
        
        splice_read = str(raw_read)[4:-1]
        
        if len(splice_read) > 0:
            dec_val = int(splice_read, base=16)
            dec_val_list.append(dec_val)
        
        else: 
            pass 

        if len(dec_val_list) > 8: 
            spliced_list = dec_val_list.sort()[0:3]
            bot_average = sum(spliced_list)/len(spliced_list)

            if bot_average > 15: 
                print("LOW DENSITY BLOCK")
            else: 
                print("HIGH DENSITY BLOCK")
        time.sleep(1)


