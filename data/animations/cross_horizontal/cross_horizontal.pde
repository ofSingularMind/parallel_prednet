float window_scale = 1.0;
float posX;
float posY;
float size; // Size of the cross
float thickness; // Thickness of the cross
float speed; // Speed of horizontal movement
int frame_rate = 1000; //<>//
boolean save_gif = true;
boolean save_frames = true;
boolean rand_background = true;
int num_frames = frame_rate / 2;
PImage[] images = new PImage[num_frames];

import gifAnimation.*;

GifMaker gifExport;


void setup() {
  size(50, 50); // Set the size of the window
  
  size = height / 4; // Initial size of the cross
  speed = width / 10; // Initial speed of horizontal movement
  posX = width / 2; // Initial X position
  posY = height / 2; // Initial Y position
  fill(0); // Set the fill color to black
  strokeWeight(3);  // Beastly
  
  frameRate(frame_rate);
  if (save_gif == true) {
    gifExport = new GifMaker(this, "cross_horizontal.gif");
    gifExport.setRepeat(0); // make it an "endless" animation
    gifExport.setTransparent(255); // make white the transparent color -- match browser bg color
    gifExport.setDelay(1000/frame_rate);  //12fps in ms
  }
  for (int i = 0; i < num_frames; i++) {
    images[i] = loadImage("/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/50x100/" + nf(i%1000, 3) + ".png");
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
  posX += speed;
  if (posX - size / 2 > width) { // Reset position when it goes beyond the screen
    posX = -size / 2;
  }

  pushMatrix(); // Save the current drawing style settings and transformations
  translate(posX, posY); // Move the origin to the new position

  // Draw the cross lines
  line(-size / 2, 0, size / 2, 0); // Horizontal line
  line(0, -size / 2, 0, size / 2); // Vertical line

  popMatrix(); // Restore the original drawing style settings and transformations

  
  if (save_gif == true) {
    gifExport.addFrame();
  
    if (frameCount == num_frames) {
      gifExport.finish();
      exit();
    }
  }
  
  if (save_frames == true) {
    saveFrame("frames/cross_horizontal/###.png");
    if (frameCount == num_frames) {
      exit();
    }
  }
}
