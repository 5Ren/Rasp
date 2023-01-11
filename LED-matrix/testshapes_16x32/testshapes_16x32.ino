#include <RGBmatrixPanel.h>

#define CLK  8
#define OE   9
#define LAT 10
#define A   A0
#define B   A1
#define C   A2

RGBmatrixPanel matrix(A, B, C, CLK, LAT, OE, false);

void setup() {

  matrix.begin();

  // fill the screen with black
  matrix.fillScreen(matrix.Color333(0, 0, 0));

  // draw some text!
  matrix.setCursor(1, 0);  // start at top left, with one pixel of spacing
  matrix.setTextSize(1);   // size 1 == 8 pixels high

  // print each letter with a rainbow color
  matrix.setTextColor(matrix.Color333(0,7,0));
  matrix.print('Y');
  matrix.setTextColor(matrix.Color333(0,7,0));
  matrix.print('a');
  matrix.setTextColor(matrix.Color333(0,7,0));
  matrix.print('m');
  matrix.setTextColor(matrix.Color333(0,7,0));
  matrix.print('a');
  matrix.setTextColor(matrix.Color333(0,7,0));
  matrix.print('-');

  matrix.setCursor(1, 9);  // next line
  matrix.setTextColor(matrix.Color333(0,7,0));
  matrix.print('L');
  matrix.setTextColor(matrix.Color333(0,7,0));
  matrix.print('a');
  matrix.setTextColor(matrix.Color333(0,7,0));
  matrix.print('b');
  matrix.setTextColor(matrix.Color333(0,7,0));
  matrix.print('.');
  matrix.setTextColor(matrix.Color333(0,7,0));
  matrix.print(' ');

  // whew!
}

void loop() {
}