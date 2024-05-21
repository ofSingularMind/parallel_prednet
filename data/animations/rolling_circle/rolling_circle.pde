float window_scale = 1.0;
float posX;
float posY;
float diameter; // Diameter of the circle
float angle = 0; // Initial rotation angle
float rotationRate = PI/10; // Rotation rate in radians per frame
float speed; // Speed of horizontal movement
int frame_rate = 100; //<>//
boolean save_gif = false;
boolean save_frames = true;
boolean rand_background = true;
int num_frames = frame_rate * 10;
PImage[] images = new PImage[num_frames];

import gifAnimation.*;

GifMaker gifExport;


void setup() {
  size(100, 50); // Set the size of the window
  diameter = height / 2; // Initial diameter of the circle
  speed = rotationRate * (diameter/2); // Initial speed of horizontal movement
  posX = width / 4; // Initial X position
  posY = 0.9*height - diameter/2; // Initial Y position
  fill(255);
  frameRate(frame_rate);
  if (save_gif == true) {
    gifExport = new GifMaker(this, "multi_rolling_circle.gif");
    gifExport.setRepeat(0); // make it an "endless" animation
    gifExport.setTransparent(255); // make white the transparent color -- match browser bg color
    gifExport.setDelay(1000/frame_rate);  //12fps in ms
  }
  strokeWeight(2);
  for (int i = 0; i < num_frames; i++) {
    images[i] = loadImage("/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/50x100/" + nf(i%1000, 3) + ".png");
  }
}

void draw() {
  if (frameCount % 100 == 0) {
    // diameter = height/random(1.1, 5);
    // speed = rotationRate * (diameter/2); // Speed of horizontal movement
    // posY = 0.9*height - diameter/2; // Y position
  }
  
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
  if (posX - diameter / 2 > width) { // Reset position when it goes beyond the screen
    posX = -diameter / 2;
  }

  pushMatrix(); // Save the current drawing style settings and transformations
  translate(posX, posY); // Move the origin to the new position
  rotate(angle); // Rotate by the specified angle

  // Draw the circle
  ellipse(0, 0, diameter, diameter);

  // Draw the lines
  line(-diameter / 2, 0, diameter / 2, 0); // Horizontal line
  line(0, -diameter / 2, 0, diameter / 2); // Vertical line

  popMatrix(); // Restore the original drawing style settings and transformations
  
  angle += rotationRate; // Update the rotation angle for the next frame
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
    saveFrame("frames/single_rolling_circle/###.png");
    if (frameCount == num_frames) {
      exit();
    }
  }
}
