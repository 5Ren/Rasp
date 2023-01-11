
#define PIN_A   18
#define PIN_B   19
#define PIN_C   21
#define PIN_D   22
#define PIN_R1  16
#define PIN_G1  26
#define PIN_B1  17
#define PIN_CLK 2
#define PIN_OE  23
#define PIN_LAT 5

void setup() 
{

  pinMode( PIN_A, OUTPUT );
  pinMode( PIN_B, OUTPUT );
  pinMode( PIN_C, OUTPUT );
  pinMode( PIN_D, OUTPUT );
  pinMode( PIN_R1, OUTPUT );
  pinMode( PIN_G1, OUTPUT );
  pinMode( PIN_B1, OUTPUT );
  pinMode( PIN_CLK, OUTPUT );
  pinMode( PIN_OE, OUTPUT );
  pinMode( PIN_LAT, OUTPUT );

  digitalWrite( PIN_A, LOW );
  digitalWrite( PIN_B, LOW );
  digitalWrite( PIN_C, LOW );
  digitalWrite( PIN_D, LOW );
  digitalWrite( PIN_R1, LOW );
  digitalWrite( PIN_G1, LOW );
  digitalWrite( PIN_B1, LOW );
  digitalWrite( PIN_LAT, LOW );

  // OE のみ HIGH で OFF になる
  digitalWrite( PIN_OE,  HIGH );
  digitalWrite( PIN_CLK, LOW );
}

void loop() 
{
  int i = 0;
  char d = 0;

  for ( d = 0; d < 16; d++ ){

    // ラッチはこのタイミングで HIGH にする必要があるぽい
    digitalWrite( PIN_LAT, HIGH ); 

    // (R=1,G=1,B=1) 
    digitalWrite( PIN_R1, HIGH );
    digitalWrite( PIN_G1, HIGH );
    digitalWrite( PIN_B1, HIGH );

    // R2,G2,B2 も結線できている場合
    // digitalWrite( PIN_R2, HIGH );
    // digitalWrite( PIN_G2, HIGH );
    // digitalWrite( PIN_B2, HIGH );

    // RGB データ (R=1,G=1,B=1) を 64ibt 分転送する
    for ( i = 0; i < 64; i++ ){
      digitalWrite( PIN_CLK, HIGH );
      digitalWrite( PIN_CLK, LOW );
    }

    // LED を消灯する (データ転送時のちらつきを防ぐ)
    digitalWrite( PIN_OE, HIGH );

    // LED パネルの光らせる列を 4bit で指定する
    digitalWrite( PIN_A, ( d & 1 ) );
    digitalWrite( PIN_B, ( d & 2 ) >> 1);
    digitalWrite( PIN_C, ( d & 4 ) >> 2);
    digitalWrite( PIN_D, ( d & 8 ) >> 3);

    // 4bit で指定した列に RGB データを転送する
    digitalWrite( PIN_LAT, LOW );

    // LED を点灯する
    digitalWrite( PIN_OE,  LOW );

  }
}
