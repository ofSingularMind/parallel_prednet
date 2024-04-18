float window_scale = 1.0;
float posX;
float posY;
float theta = 0; // Rotation angle of the ellipse
float diameter; // diameter of the ellipse
float thickness = 3; // Thickness of the ellipse
color e_color = color(0, 0, 0); // Color of the ellipse
float speed; // Speed of vertical movement
int frame_rate = 1000; //<>//
boolean save_gif = true;
boolean save_frames = false;
boolean rand_background = true;
int num_frames = 150;
PImage[] images = new PImage[num_frames];
boolean rand_size = true;
boolean rand_color = true;
boolean rand_thickness = true;
boolean rand_rotation = true;
boolean rand_occlusions = true;
int num_occlusions = 6;
float[] occlusionX = new float[num_occlusions];
float[] occlusionY = new float[num_occlusions];
float[] occlusionWidth = new float[num_occlusions];
float[] occlusionHeight = new float[num_occlusions];
float[] occ_rot = new float[num_occlusions];
color[] occ_colors = new color[num_occlusions];
int randomizationRate = 40;
int h = 50;
int w = 50;

import gifAnimation.*;

GifMaker gifExport;

public void settings() {
  size(w, h); // Set the size of the window
}

void setup() {
  //size(w, h); // Set the size of the window
  
  diameter = height / 3; // Initial length of the ellipse
  speed = width / 10; // Initial speed of vertical movement
  posX = width / 2; // Initial X position
  posY = 0; // Initial Y position
  fill(e_color); // Set the inital fill color to black
  strokeWeight(thickness);  // Set the initial stroke thickness
  rectMode(CENTER);
  
  frameRate(frame_rate);
  if (save_gif == true) {
    gifExport = new GifMaker(this, "general_ellipse_vertical.gif");
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
    diameter = random(height / 3, height / 1.2);
  }

  // Adjust color randomly
  if ((rand_color) && (frameCount % randomizationRate == 0)) {
    e_color = color(random(255), random(255), random(255));
    fill(e_color);
  }

  // Adjust rotation randomly
  if ((rand_rotation) && (frameCount % randomizationRate == 0)) {
    theta = random(0, 2 * PI);
  }

  // Adjust thickness randomly
  if ((rand_thickness) && (frameCount % randomizationRate == 0)) {
    thickness = random(1, 5);
    strokeWeight(thickness);
  }

  // Update position
  posY += speed;
  if (posY - diameter / 2 > height) { // Reset position when it goes beyond the screen
    posY = -diameter / 2;
  }

  pushMatrix(); // Save the current drawing style settings and transformations
  translate(posX, posY); // Move the origin to the new position
  rotate(theta); // Rotate the ellipse

  // Draw the circle
  ellipse(0, 0, diameter, diameter/2);

  popMatrix(); // Restore the original drawing style settings and transformations

  // Draw the randomly rotated and sized rectangular occlusions, static in the scene
  if ((rand_occlusions) && (frameCount == 1)) {
    float sw = 1;
      for (int i = 0; i < num_occlusions; i++) {
        occlusionX[i] = (i%(num_occlusions/2))*width/4-width/4;//random(-width/2, width/2);
        if (i == num_occlusions/2) {sw = -1;}
        occlusionY[i] = sw*occlusionX[i];//random(-height/2, height/2);
        occlusionWidth[i] = width/14;//random(width/16, width/8);
        occlusionHeight[i] = height/2;//random(height/6, height/2);
        occ_colors[i] = color(random(255), random(255), random(255));
        if (i < num_occlusions/2) {occ_rot[i] = PI/2;} else {occ_rot[i] = 0;}//(i%(num_occlusions/2))*PI/3-PI/3;//random(PI/5, PI/3);
      }
  }

  if (rand_occlusions) {
    for (int i = 0; i < num_occlusions; i++) {
      pushMatrix();
      translate(width/2, height/2);
      rotate(occ_rot[i]);
      occ_colors[i] = color(random(255), random(255), random(255));
      fill(occ_colors[i]);
      strokeWeight(1);
      rect(occlusionX[i], occlusionY[i], occlusionWidth[i], occlusionHeight[i]);
      popMatrix();
    }
    fill(e_color); // Reset fill color
    strokeWeight(thickness);  // Reset stroke thickness
  }


  
  if (save_gif == true) {
    gifExport.addFrame();
  
    if (frameCount == num_frames) {
      gifExport.finish();
      exit();
    }
  }
  
  if (save_frames == true) {
    saveFrame("frames/general_ellipse_vertical/###.png");
    if (frameCount == num_frames) {
      exit();
    }
  }
}
