#ble-serial -s 19b10000-e8f2-537e-4f6c-d104768a1214 -t 60 -v
import asyncio, logging
from ble_serial.bluetooth.ble_interface import BLE_interface

def receive_callback(value: bytes):
    print("Received:", value)

async def hello_sender(ble: BLE_interface):
    print("Sending 0...")
    ble.queue_send(b"0\n")
    await asyncio.sleep(0.1)
    print("Sent")
    #print("break")
    #output_0 = bytes(0, 'utf-8')

async def main():
    # None uses default/autodetection, insert values if needed
    ADAPTER = None
    SERVICE_UUID = SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
    WRITE_UUID = None
    READ_UUID = None
    DEVICE = "84:cc:a8:2e:93:da"

    ble = BLE_interface(ADAPTER, SERVICE_UUID)
    ble.set_receiver(receive_callback)

    try:
        await ble.connect(DEVICE, "public", 10.0)
        await ble.setup_chars(WRITE_UUID, READ_UUID, "rw")
        #await asyncio.gather(ble.send_loop(), hello_sender(ble))
        for i in range(10):
            ble.queue_send(b"0")
            ble._send_queue()
            await asyncio.sleep(0.1)
        print("connected")
    finally:
        print("Starting Disconnect")
        await ble.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())