float window_scale = 1.0;
float posX;
float posY;
float sq_y;
float side; // side length of the square
float angle = 0; // Initial rotation angle
int frame_rate = 100;
float rotationRate = PI/10; // Rotation rate in radians per frame
float speed; // Speed of horizontal movement
float prevAngle = 0;
Point[] points = new Point[1000];
int numPoints = 0;
color c;
boolean save_gif = true;
boolean save_frames = false;
boolean rand_background = true;
int num_frames = 100;
PImage[] images = new PImage[num_frames];

import gifAnimation.*;

GifMaker gifExport;

void setup() {
  size(100, 50); // Set the size of the window
  posX = width / 4; // Initial X position
  posY = height / 2; // Initial Y position
  side = height / 2; // Initial side length of the square
  // noFill(); // No fill for the square
  fill(255);
  frameRate(frame_rate);
  if (save_gif == true) {
    gifExport = new GifMaker(this, "single_rolling_square.gif");
    gifExport.setRepeat(0); // make it an "endless" animation
    gifExport.setTransparent(255); // make white the transparent color -- match browser bg color
    gifExport.setDelay(1000/frame_rate);  //12fps in ms
  }
  rectMode(CENTER);
  strokeWeight(2);
  points[numPoints] = new Point(posX+side/2, 0.9*height);
  numPoints += 1;

  for (int i = 0; i < num_frames; i++) {
    images[i] = loadImage("/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/50x100/" + nf(i, 3) + ".png");
  }
}

void draw() {
  if (frameCount % (100) == 0) {
    //side = height/random(1.5,5);
    //rotationRate = speed/(side/2); // Rotation rate in radians per frame
  }

  // Speed of horizontal movement
  speed = rotationRate*side*sqrt(2)/2*cos(PI/4 - angle%(PI/2));
  
  // Set the background to random pixels or white:
  if (rand_background) {
  image(images[frameCount-1], 0, 0, width, height);
  } else {
    background(255);
  }

  // Draw ground line
  line(0, 0.9*height, width, 0.9*height);
  
  // Update position
  posX += speed;
  if (posX - side / 2 > width) { // Reset position when it goes beyond the screen
    posX = -side / 2;
  }
  float A = (side/2) * (sqrt(2) - 1);
  float B = 2;
  posY = 0.9*height - side/2 - abs(A*sin(B*angle));

  // Add the new point to the array
  if (angle - prevAngle > PI/2) {
    prevAngle = angle;
    points[numPoints] = new Point(posX+side/2, 0.9*height);
    numPoints += 1;
  }
  // Draw the points
  for (int i = 0; i < numPoints; i++) {
    //points[i].display();
  }
  
  // Save the current drawing style settings and transformations
  pushMatrix();
  
  // Translate origin
  translate(posX, posY); // Move the origin to the new position
  rotate(angle); // Rotate by the specified angle

  // Draw the square
  square(0, 0, side);

  // Restore the original drawing style settings and transformations
  popMatrix(); 

  
  // Update the rotation angle for the next frame
  angle += rotationRate;
  if (angle > 2*PI) {
    // Reset the angle when it goes beyond 2*PI
    angle -= 2*PI;
  }

  if (save_gif == true) {
    gifExport.addFrame();
  
    if (frameCount == num_frames) {
      gifExport.finish();
      exit();
    }
  }

  if (save_frames == true) {
    saveFrame("frames/single_rolling_square/###.png");
    if (frameCount == num_frames) {
      exit();
    }
  }

}

class Point {
  float x, y;

  Point(float x, float y) {
    this.x = x;
    this.y = y;
  }

  void display() {
    c = color(255, 0, 0);
    fill(c);
    circle(x, y, 10);
    c = color(255);
    fill(c);
  }
}
