import time
import serial


# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial("COM9", 9600)

ser.close()
ser.open()

timeout = ser.timeout
ser.timeout = 2
dec_val_list = []

writeMode = 1

print('Enter your commands below.\r\nInsert "exit" to leave the application.')
data_in = 1
ser.write(b"20")
time.sleep(3)
ser.write(b"21")
time.sleep(3)
ser.write(b"300")
ser.write(b"310")
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
            try: 
                dec_val = int(splice_read, base=16)
                dec_val_list.append(dec_val)
                print(dec_val)
            except: 
                pass
            time.sleep(0.1)
        
        else: 
            pass 
        
        print(dec_val_list)

        if len(dec_val_list) > 4: 
            bot_average = sum(dec_val_list)/len(dec_val_list)
            print(bot_average)
            if bot_average > 40: 
                print("HIGH DENSITY BLOCK")
                ser.write(b"310")
                ser.write(b"301")
                
            else: 
                print("LOW DENSITY BLOCK")
                ser.write(b"300")
                ser.write(b"311")

            break

        time.sleep(0.1)


