float squareSize = 50; // Size of the square
float angle = 0; // Current rotation angle of the square
float posX = 0; // Current x position of the square
float speed = 2; // Speed at which the square moves

void setup() {
  size(800, 200);
  rectMode(CORNERS); // Draw rectangles using corner coordinates
}

void draw() {
  background(255); // Clear the screen with a white background
  
  // Draw the static line
  line(0, height - squareSize / 2, width, height - squareSize / 2);
  
  // Calculate the new position and rotation angle of the square
  posX += speed;
  angle += radians(speed) * (360 / (2 * PI * (squareSize / 2)));
  
  // Wrap the square to the left side of the screen if it goes off the right edge
  if (posX > width + squareSize) {
    posX = -squareSize;
  }
  
  // Save the current transformation matrix
  pushMatrix();
  
  // Move the origin to the point around which we want to rotate
  translate(posX, height - squareSize / 2);
  
  // Rotate the square
  rotate(angle);
  
  // Draw the square
  rect(0, 0, -squareSize, -squareSize);
  
  // Restore the original transformation matrix
  popMatrix();
}
