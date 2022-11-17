#include <ArduinoBLE.h>


const char* nameOfPeripheral = "M201_BLE_DEVICE"; 

// Device name

const char* uuidOfService = "0000ffe1-0000-1000-8000-00805f9b34fb";
const char* uuidOfRxChar = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";
const char* uuidOfTxChar = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";

// BLE Service

BLEService receiverService(uuidOfService); 


// Setup the incoming data characteristic (RX)

const int RX_BUFFER_SIZE = 256; 
bool      RX_BUFFER_FIXED_LENGTH = false; 

static bool      readNotWrite = true; 



BLECharacteristic rxChar(uuidOfRxChar, BLEWriteWithoutResponse | BLEWrite, RX_BUFFER_SIZE, RX_BUFFER_FIXED_LENGTH); 
BLEByteCharacteristic txChar(uuidOfTxChar, BLERead | BLENotify | BLEBroadcast);


void setup() {
  // put your setup code here, to run once:

  Serial.begin(9600); 
  while(!Serial);

  pinMode(LED_BUILTIN, OUTPUT); 

  startBLE(); 

  BLE.setLocalName(nameOfPeripheral); 
  BLE.setAdvertisedService(receiverService); 
  receiverService.addCharacteristic(rxChar);
  receiverService.addCharacteristic(txChar);
  BLE.addService(receiverService);

  BLE.setEventHandler(BLEConnected, onBLEConnected);
  BLE.setEventHandler(BLEDisconnected, onBLEDisconnected);
  
  rxChar.setEventHandler(BLEWritten, onRxCharValueUpdate);

  BLE.advertise();


  Serial.println(F("Bluetooth device active, waiting for connections..."));

  

}

void loop() {
  // put your main code here, to run repeatedly:
  byte data; 
  char response[33];
  byte charPos = 0; 

  byte test_send = "12345"; 

  BLEDevice central = BLE.central();
  
  if (central)
  {

    while(central.connected()) {
    connectedLight(); 
    
      if (readNotWrite){
        byte tmp[256]; 

        String input_str = "";

        int dataLength = rxChar.readValue(tmp, 256);

        if (dataLength > 0 ){
          for(int i = 0; i < dataLength; i++) { 
            input_str += (char)tmp[i];
          }

        if (input_str == "WRITE"){
          Serial.println(F("SWITCHING TO WRITE MODE"));
          disconnectedLight(); 
          delay(1000);
          connectedLight();
          readNotWrite = false; 
         }         
        }        
      }

      else{
        int dataLength = strlen(test_send);  
        txChar.writeValue((byte)test_send);      
      }   
    }
    disconnectedLight(); 
  }

  else {
    disconnectedLight(); 
  }
}

void startBLE() {
  if (!BLE.begin())
  {
    Serial.println(F("starting BLE failed!"));
    while (1);
  }
}

void onRxCharValueUpdate(BLEDevice central, BLECharacteristic characteristic) {
  // central wrote new value to characteristic, update LED
  Serial.print(F("Characteristic event, read: "));
  byte tmp[256];
  int dataLength = rxChar.readValue(tmp, 256);

  for(int i = 0; i < dataLength; i++) {
    Serial.print((char)tmp[i]);
  }
  
  Serial.println();
}


void onBLEConnected(BLEDevice central) {
  Serial.print("Connected event, central: ");
  Serial.println(central.address());
  connectedLight();
}

void onBLEDisconnected(BLEDevice central) {
  Serial.print("Disconnected event, central: ");
  Serial.println(central.address());
  disconnectedLight();
}

/*
 * LEDS
 */
void connectedLight() {
  digitalWrite(LED_BUILTIN, HIGH);

}

void disconnectedLight() {
 digitalWrite(LED_BUILTIN, LOW);
}

