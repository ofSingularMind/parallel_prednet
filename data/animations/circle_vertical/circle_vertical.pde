float window_scale = 1.0;
float posX;
float posY;
float diameter; // diameter of the circle
float speed; // Speed of vertical movement
int frame_rate = 1000; 
boolean save_gif = false;
boolean save_frames = false;
boolean rand_background = true;
int num_frames = 1000;
PImage[] images = new PImage[num_frames];

import gifAnimation.*;

GifMaker gifExport;

void setup() {
  size(50, 50); // Set the size of the window
  
  diameter = height / 4; // Initial length of the circle
  speed = width / 20; // Initial speed of vertical movement
  posX = width / 2; // Initial X position
  posY = height / 2; // Initial Y position
  fill(0); // Set the inital fill color to black
  strokeWeight(3);  // Set the initial stroke thickness
  rectMode(CENTER);
  
  frameRate(frame_rate);
  if (save_gif == true) {
    gifExport = new GifMaker(this, "circle_vertical.gif");
    gifExport.setRepeat(0); // make it an "endless" animation
    gifExport.setTransparent(255); // make white the transparent color -- match browser bg color
    gifExport.setDelay(1000/frame_rate);  //12fps in ms
  }
  for (int i = 0; i < num_frames; i++) {
    images[i] = loadImage("/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/50x50/" + nf(i%1000, 3) + ".png");
  }
}

void draw() {
  // Set the background to random pixels or white:
  if (rand_background) {
    image(images[frameCount-1], 0, 0, width, height);
  } else {
    background(255);
  }

  // Update position
  posY += speed;
  if (posY - diameter / 2 > width) { // Reset position when it goes beyond the screen
    posY = -diameter / 2;
  }

  pushMatrix(); // Save the current drawing style settings and transformations
  translate(posX, posY); // Move the origin to the new position

  // Draw the circle
  circle(0, 0, diameter);

  popMatrix(); // Restore the original drawing style settings and transformations

  
  if (save_gif == true) {
    gifExport.addFrame();
  
    if (frameCount == num_frames) {
      gifExport.finish();
      exit();
    }
  }
  
  if (save_frames == true) {
    saveFrame("frames/circle_vertical/###.png");
    if (frameCount == num_frames) {
      exit();
    }
  }
}
