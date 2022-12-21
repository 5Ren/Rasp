//www.elegoo.com
//2016.12.9

#include <LiquidCrystal.h>
int tempPin = 0;
//                BS  E  D4 D5  D6 D7
LiquidCrystal lcd(4, 5, 6, 7, 8, 9);
void setup() {
  Serial.begin(9600);
  lcd.begin(16, 2);
  analogWrite(3, 50);
}

void loop() {
  lcd.setCursor(0, 0);
  lcd.print("Laser is Active!");

  String putstring;
  String message ="Please pay attention. ";
  String blank = "                ";
  message = blank + message;

  Serial.println(message);
  static int row = message.length();
  Serial.println(row);

  // lcd.setCursor(0, 1);  
  for (int i = 1; i <= row; i++) {
    lcd.setCursor(0, 1); 
    lcd.print(message.substring(i, 16 + i));
    delay(700);
  }
}

