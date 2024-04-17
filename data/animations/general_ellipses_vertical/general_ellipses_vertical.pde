float window_scale = 1.0;
float posX;
float posY;
float theta = 0; // Rotation angle of the ellipse
float diameter; // diameter of the ellipse
float thickness = 3; // Thickness of the ellipse
PVector e_color = new PVector(0, 0, 0); // Color of the ellipse
float speed; // Speed of vertical movement
int frame_rate = 1000; //<>//
boolean save_gif = true;
boolean save_frames = false;
boolean rand_background = true;
int num_frames = 100;
PImage[] images = new PImage[num_frames];
boolean rand_size = true;
boolean rand_color = true;
boolean rand_thickness = true;
boolean rand_rotation = true;
boolean rand_occlusions = true;
int randomizationRate = 4;
int h = 50;
int w = 50;

import gifAnimation.*;

GifMaker gifExport;

public void settings() {
  size(w, h); // Set the size of the window
}

void setup() {
  //size(w, h); // Set the size of the window
  
  diameter = height / 4; // Initial length of the ellipse
  speed = width / 20; // Initial speed of vertical movement
  posX = width / 2; // Initial X position
  posY = height / 2; // Initial Y position
  fill(e_color.x, e_color.y, e_color.z); // Set the inital fill color to black
  strokeWeight(thickness);  // Set the initial stroke thickness
  rectMode(CENTER);
  
  frameRate(frame_rate);
  if (save_gif == true) {
    gifExport = new GifMaker(this, "general_ellipses_vertical.gif");
    gifExport.setRepeat(0); // make it an "endless" animation
    gifExport.setTransparent(255); // make white the transparent color -- match browser bg color
    gifExport.setDelay(1000/frame_rate);  //12fps in ms
  }
  for (int i = 0; i < num_frames; i++) {
    images[i] = loadImage("/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/"+nf(h)+"x"+nf(w)+"/" + nf(i%1000, 3) + ".png");
  }
}

void draw() {
  // Set the background to random pixels or white:
  if (rand_background) {
    image(images[frameCount-1], 0, 0, width, height);
  } else {
    background(255);
  }

  // Adjust size randomly
  if ((rand_size) && (frameCount % randomizationRate == 0)) {
    diameter = random(height / 6, height / 1.5);
  }

  // Adjust color randomly
  if ((rand_color) && (frameCount % randomizationRate == 1)) {
    e_color = new PVector(random(255), random(255), random(255));
    fill(e_color.x, e_color.y, e_color.z);
  }

  // Adjust rotation randomly
  if ((rand_size) && (frameCount % randomizationRate == 2)) {
    theta = random(0, 2 * PI);
  }

  // Adjust thickness randomly
  if ((rand_size) && (frameCount % randomizationRate == 3)) {
    thickness = random(1, 10);
    strokeWeight(thickness);
  }

  // Update position
  posY += speed;
  if (posY - diameter / 2 > width) { // Reset position when it goes beyond the screen
    posY = -diameter / 2;
  }

  pushMatrix(); // Save the current drawing style settings and transformations
  translate(posX, posY); // Move the origin to the new position
  rotate(theta); // Rotate the ellipse

  // Draw the circle
  ellipse(0, 0, diameter, diameter/2);

  // Draw the randomly rotated and sized rectangular occlusions
  if ((rand_occlusions)) {
    for (int i = 0; i < 3; i++) {
      float occlusionX = random(-diameter/2, diameter/2);
      float occlusionY = random(-diameter/2, diameter/2);
      float occlusionWidth = random(0, diameter/2);
      float occlusionHeight = random(0, diameter/2);
      pushMatrix();
      rotate(random(0, 2 * PI));
      fill(random(255), random(255), random(255));
      strokeWeight(1);
      rect(occlusionX, occlusionY, occlusionWidth, occlusionHeight);
      popMatrix();
    }
    fill(e_color.x, e_color.y, e_color.z); // Reset fill color
    strokeWeight(thickness);  // Reset stroke thickness
  }

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
