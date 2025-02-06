int cw_pin = 10;
int ccw_pin = 11;
int delay_ms = 50;

void setup() {
  Serial.begin(9600);
  pinMode(cw_pin, OUTPUT);
  pinMode(ccw_pin, OUTPUT);

}

void loop() {
  char input_data[10];
  char deg_data[10];
  int deg_pulse
  int data_len;

  if (Serial.available()) {
    // シリアル通信で1行（セミコロンまで）読み込む
    input_data = Serial.readStringUntil(';');
    Serial.println(input_data[]);
    data_len = input_data.length();

    // インプットから角度をとってくる
    for (int i = 0; i < data_len; i++) {
      deg_data[i] = input_data[(i + 1)];
    }
    // 文字列を整数に変換
    deg_pulse = deg_data.toInt()

    if (deg_data > 0) {
      switch (input_data[0]) {
        case 'c':
          for (int deg = 0; deg < deg_pulse; deg++) {
            digitalWrite(cw_pin, HIGH);
            delay(delay_ms);
            digitalWrite(cw_pin, LOW);
            delay(delay_ms);
          }
          break;

        case 'w':
          for (int deg = 0; deg < deg_pulse; deg++) {
            digitalWrite(ccw_pin, HIGH);
            delay(delay_ms);
            digitalWrite(ccw_pin, LOW);
            delay(delay_ms);
          }
          break;

        default:
          digitalWrite(ccw_pin, LOW);
          digitalWrite(cw_pin, LOW);
          break;
      }
    }
  }
}