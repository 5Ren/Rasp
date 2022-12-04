int cwPin = 10;
int ccwPin = 11;
int delayMicroSec = 1000;

void setup() {
  Serial.begin(19200);
  pinMode(cwPin, OUTPUT);
  pinMode(ccwPin, OUTPUT);

}

void loop() {

  if (Serial.available()) {
    // シリアル通信で1行（セミコロンまで）読み込む
    String inputData = Serial.readStringUntil(';');
    //Serial.println(inputData);

    // インプットから角度をとってくる
    String degPulseStr = inputData.substring(1);
    // 文字列を整数に変換
    long degPulse = degPulseStr.toInt();

    //  
    if (degPulse > 0) {
      
      int targetPin = 0; 
      switch (inputData.charAt(0)) {
        case 'c':
          targetPin = cwPin;
          break;
        case 'w':
          targetPin = ccwPin;
          break;
      }
      
      for (int pulse = 0; pulse < degPulse; pulse++) {
        digitalWrite(targetPin, HIGH);
        delayMicroseconds(delayMicroSec);
        digitalWrite(targetPin, LOW);
        delayMicroseconds(delayMicroSec);
      }

    }

    digitalWrite(cwPin, LOW);
    digitalWrite(ccwPin, LOW);
    
    Serial.println("stop");

  }
}