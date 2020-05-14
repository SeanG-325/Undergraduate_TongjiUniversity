/*
* Gardening.c
* Sean's Gardening Tongji University
*
* Copyright (c) 2018
* Author      : Sean
* Create Time:  2017.10
* Change Log :
*
* The MIT License (MIT)
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/
#include <Wire.h>
#include "SeeedOLED.h"
#include <EEPROM.h>
#include "DHT.h"
#include <TimerOne.h>
#include "Arduino.h"
#include "SI114X.h"
#include <SoftwareSerial.h>
#include <WiFly.h>
#include "HTTPClient.h"
#include <string.h>
#include <SD.h>
#include <SPI.h>
#include "TFTv2.h"
/*
enum Status
{
  Standby = 0,
  Warning = 1,
  Setting = 2,
  Watering = 3,
};
*/
//typedef enum Status Systemstatus;
//Systemstatus WorkingStatus;

/*
enum EncoderDir
{
  Anticlockwise = 0,
  Clockwise = 1,
};
*/
//typedef enum EncoderDir EncodedirStatus;
//EncodedirStatus EncoderRoateDir;


enum WarningStatus
{
  NoWarning = 0,
  AirHumitityWarning = 1,
  AirTemperWarning = 2,
  UVIndexWarning = 3,
  NoWaterWarning = 4,
};
typedef enum WarningStatus WarningStatusType;
WarningStatusType SystemWarning;

void WaterPumpOn();
void WaterPumpOff();
struct Limens
{
  unsigned char UVIndex_Limen = 9;
  unsigned char DHTHumidity_Hi = 60;
  unsigned char DHTHumidity_Low = 0;
  unsigned char DHTTemperature_Hi = 30;
  unsigned char DHTTemperature_Low = 0;
  unsigned char MoisHumidity_Limen = 0;
  int           CO2PPM_Limen = 2000;
  float         WaterVolume = 0;
};
typedef struct Limens WorkingLimens;
WorkingLimens SystemLimens;

#define DHTPIN          A6     // what pin we're connected to
#define MoisturePin     A8
#define ButtonPin       4
// Uncomment whatever type you're using!
#define DHTTYPE DHT11   // DHT 11 
//#define DHTTYPE DHT22   // DHT 22  (AM2302)
//#define DHTTYPE DHT21   // DHT 21 (AM2301)
#define EncoderPin1     2
#define EncoderPin2     3
#define WaterflowPin    2
#define RelayPin        8
#define RelayPin2       

#define OneSecond       1000
#define DataUpdateInterval 8000  // 20S
#define RelayOn         HIGH
#define RelayOff        LOW

#define NoWaterTimeOut  3        // 10s

unsigned int  uiWaterVolume = 0;
unsigned char WaterflowFlag = 0;
unsigned int  WaterflowRate = 0;  // L/Hour
unsigned int  NbTopsFan = 0;  // count the edges

unsigned char EncoderFlag = 0;
unsigned long StartTime = 0;
unsigned char ButtonFlag = 0;
signed   char LCDPage = 4;
unsigned char SwitchtoWateringFlag = 0;
unsigned char SwitchtoWarningFlag = 0;
unsigned char SwitchtoStandbyFlag = 0;
unsigned char UpdateDataFlag = 0;
unsigned char ButtonIndex = 0;
unsigned char EEPROMAddress = 0;
float Volume = 0;
unsigned long counter = 0;

SI114X SI1145 = SI114X();
DHT dht(DHTPIN, DHTTYPE);
float DHTHumidity = 0;
float DHTTemperature = 0;
float MoisHumidity = 0;
float UVIndex = 0;
//char buffer[30];



#define SSID "TP-LINK_2860"
#define KEY "656wansui"

#define SSID2      "guanenwu"
#define KEY2       "Guanshuo1997"


//#define SSID4 
//#define KEY4 

//#define SSID "Sean"
//#define KEY "guanshuo"

// WIFLY_AUTH_OPEN / WIFLY_AUTH_WPA1 / WIFLY_AUTH_WPA1_2 / WIFLY_AUTH_WPA2_PSK
#define AUTH      WIFLY_AUTH_WPA2_PSK


int Watering=0;

#define HTTP_POST_URL "http://115.159.201.24/receive_data.php?Json="
//#define HTTP_POST_URL  "http://api.yeelink.net/v1.1/device/360478/sensor/411279/datapoints"
//#define HTTP_POST_HEADERS "Accept: */*\r\nAccept-Language: zh-cn\r\nU-ApiKey: 1e1a5856ad4b999d59b1a8bc3716bc09\r\nConnection: keep-alive\r\nContent-Type: text/plain;charset:utf-8\r\n"
#define HTTP_POST_HEADERS "Connection: keep-alive\r\nAccept: */*\r\nAccept-Language: zh-cn\r\nContent-Type: application/x-www-form-urlencoded;charset:utf-8\r\n"
char ins[50] = { 0 };
// Pins' connection
// Arduino       WiFly
//  7    <---->    TX
//  8    <---->    RX
SoftwareSerial uart(A12, A14);
WiFly wifly(uart);
HTTPClient http;
char get;
//char post_data_buf[100]={0};
SoftwareSerial s_serial(A10, A11);      // TX, RX
const unsigned char cmd_get_sensor[] =
{
  0xff, 0x01, 0x86, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x79
};
unsigned char dataRevice[9];
int temperature;
int CO2PPM;
#define CO2_sensor s_serial


#define MAX_BMP         10                      // bmp file num
#define FILENAME_LEN    20                      // max file name length


const int PIN_SD_CS = 4;                        // pin of sd card

const int __Gnbmp_height = 320;                 // bmp hight
const int __Gnbmp_width  = 240;                 // bmp width

unsigned char __Gnbmp_image_offset  = 0;        // offset
int __Gnfile_num = 4;                           // num of file
char __Gsbmp_files[FILENAME_LEN] =  "back.bmp";         // add file name here
File bmpFile;

#define BUFFPIXEL       60                      // must be a divisor of 240 
#define BUFFPIXEL_X3    180                     // BUFFPIXELx3
void bmpdraw(File f, int x, int y);
boolean bmpReadHeader(File f);
uint16_t read16(File f)
{
    uint16_t d;
    uint8_t b;
    b = f.read();
    d = f.read();
    d <<= 8;
    d |= b;
    return d;
}
uint32_t read32(File f)
{
    uint32_t d;
    uint16_t b;

    b = read16(f);
    d = read16(f);
    d <<= 16;
    d |= b;
    return d;
}
  int UVIndex_Limen = 0;
  int DHTHumidity_Hi = 0;
  int DHTHumidity_Low = 0;
  int DHTTemperature_Hi = 0;
  int DHTTemperature_Low = 0;
  int MoisHumidity_Limen = 0;
  int CO2PPM_Limen = 0;
void setup()
{
  /* Init OLED */
  //Wire.begin();
  //    encoder.Timer_init();
  /* Init DHT11 */
  char ins[100]={0};
  int i=0;
  Serial.begin(9600);
  uart.listen();
  uart.begin(9600);     //WiFly UART Baud Rate: 9600
  
/*
  SeeedOled.init();  //initialze SEEED OLED display
  DDRB |= 0x21;
  PORTB |= 0x21;
  SeeedOled.clearDisplay();          //clear the screen and set start position to top left corner
  SeeedOled.setNormalDisplay();      //Set display to normal mode (i.e non-inverse mode)
  SeeedOled.setPageMode();           //Set addressing mode to Page Mode
  */
  Serial.println("DHTxx test!");
  dht.begin();
  uart.listen();
  Serial.println("Joining wifi!");
  
  if (!wifly.isAssociated(SSID)) {
    while (!wifly.join(SSID, KEY, AUTH)) {
      Serial.println("Failed to join " SSID);
      //SeeedOled.setTextXY(2, 4);
      //SeeedOled.putString("Fail to join wifi.");
      Serial.println("Wait 0.1 second and try again...");
      delay(100);
    }
    wifly.save();    // save configuration, 
  }
  wifly.sendCommand("set comm remote 0\r"); // turn off the REMOTE string so it does not interfere with the post
  //wifly.sendCommand("set comm close 0\r"); // turn off the REMOTE string so it does not interfere with the post
  //sprintf(post_data_buf, "1111-QWER-%d.%d-%d.%d-%d.%d-%d.%d-%d\r\n", (int)(DHTHumidity), (int)((int)(DHTHumidity * 100) % 100), (int)(DHTTemperature), (int)((int)(DHTTemperature * 100) % 100), (int)(MoisHumidity), (int)((int)(MoisHumidity * 100) % 100), (int)(UVIndex), (int)((int)(UVIndex * 100) % 100), CO2PPM);
      //sprintf(post_data_buf,"%d.%d",(int)(k),(int)((int)(k*100) %100));
      //Serial.println(post_data_buf);
      //strcat(key,post_data_buf);
      //Serial.println(key);
      //int Start_Time = millis(); 
      while (http.get("http://115.159.201.24/send_schedule.php?Json=1111-QWER-request\r\n", HTTP_POST_HEADERS) < 0) {
        //if(millis()-Start_Time > 5000)
        //break;
       }
       while (wifly.receive((uint8_t*)&ins[i], 1, 1000) == 1)
      {
        if(ins[i]=='*')
        {
          ins[i]=0;
          i--;
          break;
          }
        Serial.print(ins[i]);
        i++;
        }
        Serial.print("\n");
        Serial.println(ins);
       sscanf(ins,"%d %d %d %d %d %d %d",&UVIndex_Limen, &DHTHumidity_Hi, &DHTHumidity_Low, &DHTTemperature_Hi, &DHTTemperature_Low, &MoisHumidity_Limen, &CO2PPM_Limen);
       Serial.println(UVIndex_Limen);
       Serial.println(DHTTemperature_Hi);
        
  
  delay(100);
  //uart.end();
  Tft.TFTinit();
  pinMode(PIN_SD_CS,OUTPUT);
  digitalWrite(PIN_SD_CS,HIGH);
  
  Sd2Card card;
  card.init(SPI_FULL_SPEED, PIN_SD_CS); 
  if(!SD.begin(PIN_SD_CS))              
  { 
      Serial.println("failed!");
      while(1);                               // init fail, die here
  }
  Serial.println("SD OK!");
  TFT_BL_ON;
  
  Tft.drawString("Initializing...",0,10,2,WHITE);       // draw string: "hello", (0, 180), size: 3, color: CYAN
  Tft.drawString("Joining wifi...",0,85,2,WHITE);       // draw string: "hello", (0, 180), size: 3, color: CYAN
  Tft.drawString("Success.",0,160,2,WHITE);       // draw string: "hello", (0, 180), size: 3, color: CYAN
  //float k = random(10,100);
  //sprintf(post_data_buf, "%d.%d %d.%d %d.%d %d.%d\r\n", (int)(DHTHumidity),(int)((int)(DHTHumidity*100) %100,(int)(DHTTemperature)),(int)((int)(DHTTemperature*100) %100, (int)(MoisHumidity)),(int)((int)(MoisHumidity*100) %100, (int)(UVIndex),(int)((int)(UVIndex*100) %100)));
  //sprintf(post_data_buf,"%d.%d",(int)(k),(int)((int)(k*100) %100));
  //Serial.println(post_data_buf);
  //while (http.post(HTTP_POST_URL, HTTP_POST_HEADERS, post_data_buf) < 0) {
  //}   
  /* Init Button */
  //pinMode(ButtonPin, INPUT);
  //attachInterrupt(0, ButtonClick, FALLING);
  /* Init Encoder */
  //pinMode(EncoderPin1, INPUT);
  //pinMode(EncoderPin2, INPUT);
  //attachInterrupt(1, EncoderRotate, RISING);

  /* Init UV */
  while (!SI1145.Begin()) {
    Serial.print("Error!\n");
    delay(1000);
  }


  /* Init Water flow */
  pinMode(WaterflowPin, INPUT);

  /* Init Relay      */
  pinMode(RelayPin, OUTPUT);
  /* The First time power on to write the default data to EEPROM */
  EEPROM.write(EEPROMAddress, 0x00);
  EEPROM.write(++EEPROMAddress, SystemLimens.UVIndex_Limen);
  EEPROM.write(++EEPROMAddress, SystemLimens.DHTHumidity_Hi);
  EEPROM.write(++EEPROMAddress, SystemLimens.DHTHumidity_Low);
  EEPROM.write(++EEPROMAddress, SystemLimens.DHTTemperature_Hi);
  EEPROM.write(++EEPROMAddress, SystemLimens.DHTTemperature_Low);
  EEPROM.write(++EEPROMAddress, SystemLimens.MoisHumidity_Limen);
  EEPROM.write(++EEPROMAddress, ((int)(SystemLimens.WaterVolume * 100)) / 255);    /*  */
  EEPROM.write(++EEPROMAddress, ((int)(SystemLimens.WaterVolume * 100)) % 255);
  /*
  else { /* If It's the first time power on , read the last time data

  EEPROMAddress++;
  SystemLimens.UVIndex_Limen      = EEPROM.read(EEPROMAddress++);
  SystemLimens.DHTHumidity_Hi     = EEPROM.read(EEPROMAddress++);
  SystemLimens.DHTHumidity_Low    = EEPROM.read(EEPROMAddress++);
  SystemLimens.DHTTemperature_Hi  = EEPROM.read(EEPROMAddress++);
  SystemLimens.DHTTemperature_Low = EEPROM.read(EEPROMAddress++);
  SystemLimens.MoisHumidity_Limen = EEPROM.read(EEPROMAddress++);
  SystemLimens.WaterVolume =   (EEPROM.read(EEPROMAddress++)*255 + EEPROM.read(EEPROMAddress))/100.0;
  }*/

  StartTime = millis();
//  WorkingStatus = Standby;
//  SystemWarning = NoWarning;
  CO2_sensor.listen();
  CO2_sensor.begin(9600);
  StartTime=millis();
}

//unsigned long start_millis = 0;

//bool Ctrlforwater = 0;
bool dataReceive();
void loop()
{
     /*
    WaterPumpOn();
    delay(1000);
    WaterPumpOff();
    */
    Serial.print("12345\n");
    int WaterTime = 0;
    char buffer[30] = { 0 };
    char post_data_buf[100] = { 0 };
    int i = 0, at = 0, light = 0, water = 0;
    // Reading temperature or humidity takes about 250 milliseconds!
    // Sensor readings may also be up to 2 seconds 'old' (its a very slow sensor)
    //switch (WorkingStatus) {
    //case Standby:
    if (millis() - StartTime > DataUpdateInterval) 
    {
      StartTime = millis();
      DHTHumidity = dht.readHumidity();
      DHTTemperature = dht.readTemperature();
      MoisHumidity = analogRead(MoisturePin) / 7;
      UVIndex = (float)SI1145.ReadUV() / 100 + 0.5;
      CO2_sensor.listen();
      if (!dataReceive())
      {
        //ERROR or WAIT
       }

      //Serial.println("\r\n\r\nTry to post data to url - " HTTP_POST_URL);
      //Serial.println("-------------------------------");
      //float k = random(10,100);
      
      bmpFile = SD.open(__Gsbmp_files);
      if (! bmpFile)
      {
          Serial.println("didnt find image");
          while (1);
      }
      if(! bmpReadHeader(bmpFile)) 
      {
          Serial.println("bad bmp");
          return;
      }
      bmpdraw(bmpFile, 0, 0);
      bmpFile.close();

      
      if(DHTHumidity < SystemLimens.DHTHumidity_Hi && DHTHumidity > SystemLimens.DHTTemperature_Low)
      Tft.drawFloat(DHTHumidity,60,18,2,BLUE);       // draw string: "hello", (0, 180), size: 3, color: CYAN
      else
      Tft.drawFloat(DHTHumidity,60,18,2,RED);
      
      if(UVIndex > SystemLimens.UVIndex_Limen)
      Tft.drawFloat(UVIndex,60,80,2,BLUE);
      else
      Tft.drawFloat(UVIndex,60,80,2,RED);
      
      if(MoisHumidity > SystemLimens.MoisHumidity_Limen)
      Tft.drawFloat(MoisHumidity,60,142,2,BLUE);       // draw string: "hello", (0, 180), size: 3, color: CYAN
      else
      Tft.drawFloat(MoisHumidity,60,142,2,RED);
      
      if(DHTTemperature > SystemLimens.DHTTemperature_Low && DHTTemperature < SystemLimens.DHTTemperature_Hi)
      Tft.drawFloat(DHTTemperature,60,204,2,BLUE);       // draw string: "hello", (0, 180), size: 3, color: CYAN
      else
      Tft.drawFloat(DHTTemperature,60,204,2,RED);
      
      if(CO2PPM < 2000)
      Tft.drawFloat(CO2PPM,60,268,2,BLUE);       // draw string: "hello", (0, 180), size: 3, color: CYAN
      else
      Tft.drawFloat(CO2PPM,60,268,2,RED);

      
      uart.listen();
      //uart.begin(9600);
      char key[100]={0};
      strcpy(key,HTTP_POST_URL); 
      sprintf(post_data_buf, "1111-QWER-%d.%d-%d.%d-%d.%d-%d.%d-%d\r\n", (int)(DHTHumidity), (int)((int)(DHTHumidity * 100) % 100), (int)(DHTTemperature), (int)((int)(DHTTemperature * 100) % 100), (int)(MoisHumidity), (int)((int)(MoisHumidity * 100) % 100), (int)(UVIndex), (int)((int)(UVIndex * 100) % 100), CO2PPM);
      //sprintf(post_data_buf,"%d.%d",(int)(k),(int)((int)(k*100) %100));
      Serial.println(post_data_buf);
      strcat(key,post_data_buf);
      Serial.println(key);
      int Start_Time = millis(); 
      while (http.get(key, HTTP_POST_HEADERS) < 0) {
        //if(millis()-Start_Time > 5000)
        //break;
       } 
       /*
char r;
       while (wifly.receive((uint8_t *)&r, 1, 1000) == 1) {
    Serial.print(r);
  }
  */
       i=0;
      while (wifly.receive((uint8_t*)&ins[i], 1, 1000) == 1)
      {
        Serial.print(ins[i]);
        i++;
        }
        
        Serial.println(i);
       
      i = 0;
      //uart.end();
      while(!(ins[i]=='\n' && ins[i+1]=='\n'))
      {
        i++;
         //Serial.print("\n123\n");
        }
      
      //Serial.print("\n123\n");
      i+=2;
      //sscanf(ins[i], "%d %d %d", &at, &light, &water);
      //Serial.printf();
      if (at == 0)
      {
        if (light == 1)
        {
          //lighting
          
        }
        if (water == 1)
        {
          if(Watering == 0)
          {
            Watering=1;
            WaterTime=millis();
            WaterPumpOn();
            }
          if(millis()-WaterTime > 5000)
          if(Watering == 1)
          {
            Watering=0;
            WaterPumpOff();
            }
        }
      }
      else
      {
        if (MoisHumidity >100) 
        {
          MoisHumidity = 100;
         }
      }
    }
    delay(1000);
  //case Watering:
    SwitchtoWarningFlag = 0;




    
    if (digitalRead(WaterflowPin) == 1) {
      if (digitalRead(WaterflowPin) == 1) {
        if (WaterflowFlag == 0) {
          WaterflowFlag = 1;
          NbTopsFan++;
        }
      }
    }
    else {
      if (WaterflowFlag == 1) {
        WaterflowFlag = 0;
      }
      else {
      }
    }




    /*
    static char NoWaterTime = 0;
    if ((millis() - StartTime) > OneSecond) {
      WaterflowRate = (NbTopsFan * 60 / 73);
      if (((float)(NbTopsFan) / 73 / 60) < 0.005) {
        if (NoWaterTime++ >= NoWaterTimeOut) {
          NoWaterTime = 0;
          SystemWarning = NoWaterWarning;
          SwitchtoWarningFlag = 1;
          LCDPage = 3;
        }
        Volume += (float)((float)(NbTopsFan) / 73 / 60);
      }
      else {
        NoWaterTime = 0;
        SystemWarning = NoWarning;
        SwitchtoWarningFlag = 0;
        Volume += (float)((float)(NbTopsFan) / 73 / 60 + 0.005);
      }

      NbTopsFan = 0;
      sprintf(buffer, "%2d L/H", WaterflowRate);
      if ((int)((int)(Volume * 100) % 100) < 10) {
        sprintf(buffer, "%2d.0%d L", (int)(Volume), (int)((int)(Volume * 100) % 100));
      }
      else {
        sprintf(buffer, "%2d.%2d L", (int)(Volume), (int)((int)(Volume * 100) % 100));
      }
      //sprintf(buffer,"%2d.%2d L",(int)(Volume),(int)((int)(Volume*100) %100));
      //SeeedOled.setTextXY(4, 8);
      //SeeedOled.putString(buffer);


      if (Volume >= SystemLimens.WaterVolume) {
        SwitchtoStandbyFlag = 1;
      }
      //            
      // sprintf(buffer,"Press Btn toSTOP");
      // SeeedOled.setTextXY(7,0);
      // SeeedOled.putString(buffer);
      StartTime = millis();
    }
    if (SwitchtoStandbyFlag == 1) {
      Volume = 0;
      //WorkingStatus = Standby;
      //SeeedOled.clearDisplay();
      StartTime = millis();
      //LCDPage = 3;
      SwitchtoStandbyFlag = 0;
      WaterPumpOff();
      ButtonFlag = 0;
    }
    if (SwitchtoWarningFlag == 1) {
      Volume = 0;
      SwitchtoWarningFlag = 0;
      //WorkingStatus = Warning;
      StartTime = millis();
      WaterPumpOff();
      ButtonFlag = 0;
    }
    /*
    break;
  default:
    break;
    */
  }


void WaterPumpOn()
{
  digitalWrite(RelayPin, RelayOn);
}

void WaterPumpOff()
{
  digitalWrite(RelayPin, RelayOff);
}
/*
void DisplayCO2PPM(char* buffer)
{
  sprintf(buffer, "CO2");
  SeeedOled.setTextXY(1, 7);
  SeeedOled.putString(buffer);

  sprintf(buffer, "%d PPM", (int)(CO2PPM));
  SeeedOled.setTextXY(2, 6);          //Set the cursor to Xth Page, Yth Column  
  SeeedOled.putString(buffer);

  sprintf(buffer, "Safe Vaule");
  SeeedOled.setTextXY(5, 3);
  SeeedOled.putString(buffer);

  sprintf(buffer, "%2d PPM", (int)(SystemLimens.CO2PPM_Limen));
  SeeedOled.setTextXY(6, 4);

  SeeedOled.putString(buffer);

  if (digitalRead(ButtonPin) == 1) {
    Serial.println("BUTTON");
    if (EncoderFlag == 1) {
      delay(100);
      EncoderFlag = 0;
      UpdateDataFlag = 1;
      switch (EncoderRoateDir) {
      case Clockwise:
        SystemLimens.CO2PPM_Limen++;
        break;
      case Anticlockwise:
        SystemLimens.CO2PPM_Limen--;
        break;
      default:
        break;
      }
      if (SystemLimens.CO2PPM_Limen > 15) {
      SystemLimens.CO2PPM_Limen = 15;
      }
      if (SystemLimens.CO2PPM_Limen <= 0) {
      SystemLimens.CO2PPM_Limen = 0;
      }
    }
  }
}
*/


bool dataReceive(void)
{
  byte data[9];
  int i = 0;

  //transmit command data
  for (i = 0; i<sizeof(cmd_get_sensor); i++)
  {
    CO2_sensor.write(cmd_get_sensor[i]);
  }
  delay(10);
  //begin reveiceing data
  if (CO2_sensor.available())
  {
    while (CO2_sensor.available())
    {
      for (int i = 0; i<9; i++)
      {
        data[i] = CO2_sensor.read();
      }
    }
  }

  for (int j = 0; j<9; j++)
  {
    Serial.print(data[j]);
    Serial.print(" ");
  }
  Serial.println("");

  if ((i != 9) || (1 + (0xFF ^ (byte)(data[1] + data[2] + data[3] + data[4] + data[5] + data[6] + data[7]))) != data[8])
  {
    return false;
  }

  CO2PPM = (int)data[2] * 256 + (int)data[3];
  temperature = (int)data[4] - 40;

  return true;
}

void bmpdraw(File f, int x, int y)
{
    bmpFile.seek(__Gnbmp_image_offset);

    uint32_t time = millis();

    uint8_t sdbuffer[BUFFPIXEL_X3];                 // 3 * pixels to buffer

    for (int i=0; i< __Gnbmp_height; i++)
    {

        for(int j=0; j<(240/BUFFPIXEL); j++)
        {
            bmpFile.read(sdbuffer, BUFFPIXEL_X3);
            uint8_t buffidx = 0;
            int offset_x = j*BUFFPIXEL;
            
            unsigned int __color[BUFFPIXEL];
            
            for(int k=0; k<BUFFPIXEL; k++)
            {
                __color[k] = sdbuffer[buffidx+2]>>3;                        // read
                __color[k] = __color[k]<<6 | (sdbuffer[buffidx+1]>>2);      // green
                __color[k] = __color[k]<<5 | (sdbuffer[buffidx+0]>>3);      // blue
                
                buffidx += 3;
            }

            Tft.setCol(offset_x, offset_x+BUFFPIXEL);
            Tft.setPage(i, i);
            Tft.sendCMD(0x2c);                                                  
            
            TFT_DC_HIGH;
            TFT_CS_LOW;

            for(int m=0; m < BUFFPIXEL; m++)
            {
                SPI.transfer(__color[m]>>8);
                SPI.transfer(__color[m]);
            }

            TFT_CS_HIGH;
        }
        
    }
    
    Serial.print(millis() - time, DEC);
    Serial.println(" ms");
}

boolean bmpReadHeader(File f) 
{
    // read header
    uint32_t tmp;
    uint8_t bmpDepth;
    
    if (read16(f) != 0x4D42) {
        // magic bytes missing
        return false;
    }

    // read file size
    tmp = read32(f);
    Serial.print("size 0x");
    Serial.println(tmp, HEX);

    // read and ignore creator bytes
    read32(f);

    __Gnbmp_image_offset = read32(f);
    Serial.print("offset ");
    Serial.println(__Gnbmp_image_offset, DEC);

    // read DIB header
    tmp = read32(f);
    Serial.print("header size ");
    Serial.println(tmp, DEC);
    
    
    int bmp_width = read32(f);
    int bmp_height = read32(f);
    
    if(bmp_width != __Gnbmp_width || bmp_height != __Gnbmp_height)      // if image is not 320x240, return false
    {
        return false;
    }

    if (read16(f) != 1)
    return false;

    bmpDepth = read16(f);
    Serial.print("bitdepth ");
    Serial.println(bmpDepth, DEC);

    if (read32(f) != 0) {
        // compression not supported!
        return false;
    }

    Serial.print("compression ");
    Serial.println(tmp, DEC);

    return true;
}












