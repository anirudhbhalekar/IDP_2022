#ble-serial -s 19b10000-e8f2-537e-4f6c-d104768a1214 -t 60 -v
import asyncio, logging
from ble_serial.bluetooth.ble_interface import BLE_interface

def receive_callback(value: bytes):
    print("Received:", value)

async def hello_sender(ble: BLE_interface):
    while True:
        await asyncio.sleep(3.0)
        print("Sending...")
        ble.queue_send(0 + "\n")

async def main():
    # None uses default/autodetection, insert values if needed
    ADAPTER = "hci0"
    SERVICE_UUID = SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
    WRITE_UUID = None
    READ_UUID = None
    DEVICE = "84:cc:a8:2e:93:da"

    ble = BLE_interface(ADAPTER, SERVICE_UUID)
    ble.set_receiver(receive_callback)

    try:
        await ble.connect(DEVICE, "public", 30.0)
        #await ble.setup_chars(WRITE_UUID, READ_UUID, "rw")
        #await asyncio.gather(ble.send_loop(), hello_sender(ble))
        print("connected")
    finally:
        await ble.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())