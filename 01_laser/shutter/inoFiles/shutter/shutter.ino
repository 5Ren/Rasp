int shutter = 2;

void setup() {
  Serial.begin(19200);
  pinMode(shutter, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    // シリアル通信で1行（セミコロンまで）読み込む
    String inputData = Serial.readStringUntil(';');

    // 先頭を取ってくる
    String commandName = inputData.substring(1);

    switch (inputData.charAt(0)) {
      case 'a':
        if (inputData.charAt(1) == '1') {
          digitalWrite(shutter, HIGH);
          Serial.println("OPEN");
        } else if (inputData.charAt(1) == '0') {
          digitalWrite(shutter, LOW);
          Serial.println("CLOSE");
        } else {
          Serial.println("False");
        }
        break;
    }
  }
}
