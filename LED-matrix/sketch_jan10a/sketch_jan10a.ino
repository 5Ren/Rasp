
#include <DFRobot_RGBMatrix.h> // Hardware-specific library

#define CLK  8
#define OE   9
#define LAT 10
#define A     A0
#define B     A1
#define C     A2
#define D     A3
#define E     A4
#define WIDTH 64
#define HIGH  32

DFRobot_RGBMatrix matrix(A, B, C, D, E, CLK, LAT, OE, false, WIDTH, HIGH);




void setup()
{
  matrix.begin();
  delay(500);
}

void loop() {

  // draw a pixel in solid white
  matrix.drawPixel(0, 0, matrix.Color333(7, 7, 7));
  // delay(5000);

  // // fix the screen with green
  // matrix.fillRect(0, 0, matrix.width(), matrix.height(), matrix.Color333(0, 7, 0));
  // delay(5000);

  // // fix the screen with white
  // matrix.fillRect(0, 0, matrix.width(), matrix.height(), matrix.Color333(7, 7, 7));
  // delay(5000);

  // // fix the screen with red
  // matrix.fillRect(0, 0, matrix.width(), matrix.height(), matrix.Color333(7, 0, 0));
  // delay(5000);

  // // fix the screen with blue
  // matrix.fillRect(0, 0, matrix.width(), matrix.height(), matrix.Color333(0, 0, 7));
  // delay(5000);
  // // fill the screen with 'black'
  // matrix.fillScreen(matrix.Color333(0, 0, 0));

  // // draw a box in yellow
  // matrix.drawRect(0, 0, matrix.width(), matrix.height(), matrix.Color333(7, 7, 0));
  // delay(5000);

  // // draw a box in fuchsia
  // matrix.drawRect(5, 5, 53, 23, matrix.Color333(7, 0, 7));
  // delay(5000);

  // // draw a box in yellow
  // matrix.drawRect(10, 10, 43, 13, matrix.Color333(7, 1, 3));
  // delay(5000);

  // // draw an 'X' in red
  // matrix.drawLine(0, 0, matrix.width() - 1, matrix.height() - 1, matrix.Color333(7, 0, 0));
  // matrix.drawLine(matrix.width() - 1, 0, 0, matrix.height() - 1, matrix.Color333(7, 0, 0));
  // delay(5000);

  // // draw a blue circle
  // matrix.drawCircle(10, 10, 10, matrix.Color333(0, 0, 7));
  // delay(5000);

  // // fill a violet circle
  // matrix.fillCircle(40, 21, 10, matrix.Color333(7, 0, 7));
  // delay(5000);

  // // fill the screen with 'black'
  // matrix.fillScreen(matrix.Color333(0, 0, 0));

  // // draw some text!
  matrix.setTextSize(1);     // size 1 == 8 pixels high
  matrix.setTextWrap(false); // Don't wrap at end of line - will do ourselves
  matrix.setCursor(3, 0);  // start at top left, with 3 pixel of spacing
  uint8_t w = 0;
  char *str = "Welcome ToDFROBOT";
  for (w = 0; w < 10; w++) {
    matrix.setTextColor(Wheel(w));
    matrix.print(str[w]);
  }

  matrix.setCursor(13, 8);    // next line
  for (w = 10; w < 17; w++) {
    matrix.setTextColor(Wheel(w));
    matrix.print(str[w]);
  }
  matrix.println();
  matrix.setCursor(2, 16);
  matrix.setTextColor(matrix.Color333(7, 7, 7));
  matrix.println("I'm always");

  // print each letter with a rainbow color
  matrix.setCursor(3, 24);
  matrix.setTextColor(matrix.Color333(7, 0, 0));
  matrix.print('B');
  matrix.setTextColor(matrix.Color333(7, 4, 0));
  matrix.print('y');
  matrix.setTextColor(matrix.Color333(7, 7, 0));
  matrix.print(' ');
  matrix.setTextColor(matrix.Color333(4, 7, 0));
  matrix.print('U');
  matrix.setTextColor(matrix.Color333(0, 7, 0));
  matrix.print(' ');
  matrix.setTextColor(matrix.Color333(0, 7, 7));
  matrix.print("S");
  matrix.setTextColor(matrix.Color333(0, 4, 7));
  matrix.print('i');
  matrix.setTextColor(matrix.Color333(0, 0, 7));
  matrix.print('d');
  matrix.setTextColor(matrix.Color333(4, 0, 7));
  matrix.print("e");
  matrix.setTextColor(matrix.Color333(7, 0, 4));
  matrix.println("!");
  delay(50000);


}


// Input a value 0 to 24 to get a color value.
// The colours are a transition r - g - b - back to r.
uint16_t Wheel(byte WheelPos) {
  if (WheelPos < 8) {
    return matrix.Color333(7 - WheelPos, WheelPos, 0);
  } else if (WheelPos < 16) {
    WheelPos -= 8;
    return matrix.Color333(0, 7 - WheelPos, WheelPos);
  } else {
    WheelPos -= 16;
    return matrix.Color333(0, WheelPos, 7 - WheelPos);
  }
}