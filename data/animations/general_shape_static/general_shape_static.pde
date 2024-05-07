float window_scale = 1.0;
float posX;
float posY;
float theta = 0; // Rotation angle of the shape
float gamma = 0; // Rotation angle of the motion
String shape = "ellipse"; // Shape to be drawn, can be "cross", "rectangle", or "ellipse"
String mot_dir; // Motion direction, filled in automatically based on shape 
float traj_len; // Length of the trajectory
float s_len, s_wid, old_s_len; // length and width of the shape
float thickness = 3; // stroke thickness of the shape
color e_color = color(0, 0, 0); // Color of the shape
float speed; // Speed of vertical movement
int frame_rate; //<>//
int num_frames;
PImage[] images;
float d, s_wid_divisor; // Random size value
float a, b, c; // Random color values
float thk; // Random thickness value
float th; // Random rotation value
int num_occlusions = 6;
float[] occlusionX = new float[num_occlusions];
float[] occlusionY = new float[num_occlusions];
float[] occlusionWidth = new float[num_occlusions];
float[] occlusionHeight = new float[num_occlusions];
color[] occ_colors = new color[num_occlusions];
float[] occ_rot = new float[num_occlusions];
int randomizationRate;
int ws = 50;
boolean rand_size = true;
boolean rand_color = true;
boolean rand_thickness = true;
boolean rand_rotation = true;
boolean rand_occlusions = true;
String rand_background = "pixels"; // can be "pixels", "whole", "white"

boolean save_gif = false; // only set save_gif or save_frames to true, not both, or both to false
boolean save_frames = true;
boolean second_stage = true; // switches to white background and grey occlusions to sharpen up predictions

// boolean train_mode = false; // just flip this one to switch between train and test modes
// boolean test_mode = !train_mode;

boolean exec_randomize = true;
// boolean flip = false;

import gifAnimation.*;

GifMaker gifExport;

public void settings() {
  if ((save_gif == false) && (save_frames == false)) {size(500, 500);} // Set the size of the window, w, h
  else {size(ws, ws);} // Set the size of the window, w, h
}

void setup() {
  // Set the frame rate and the number of frames to be saved per mode
  if (save_gif && save_frames) {println("Error: save_gif and save_frames cannot both be true."); exit();}
  if (save_gif == true) {num_frames = 150; frame_rate = 1000;}
  else if (save_frames) {num_frames = 5000; frame_rate = 5000;}
  else {num_frames = 1000; frame_rate = 2;}
  images = new PImage[num_frames];
  
  s_len = sqrt(2) * height / 3; // Initial length of the cross
  s_wid = s_len / 4; // Initial thickness of the cross
  posX = 4 * width / 8; // Initial X position
  posY = 4 * height / 8; // Initial Y position
  fill(e_color); // Set the inital fill color to black
  stroke(0); // Set the initial stroke color to white
  strokeWeight(thickness);  // Set the initial stroke thickness
  rectMode(CENTER);
  
  frameRate(frame_rate);
  if (save_gif == true) {
    if (!second_stage) {gifExport = new GifMaker(this, "general_" + shape + "_static_1st_stage.gif");}
    else if (second_stage) {gifExport = new GifMaker(this, "general_" + shape + "_static_2nd_Stage.gif");}
    else {println("Error: stage not defined."); exit();}
    gifExport.setRepeat(0); // make it an "endless" animation
    // gifExport.setTransparent(255); // make white the transparent color -- match browser bg color
    gifExport.setDelay(1000/frame_rate);  //12fps in ms
  }
  for (int i = 0; i < num_frames; i++) {
    images[i] = loadImage("/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/"+nf(ws)+"x"+nf(ws)+"/" + nf(i%1000, 3) + ".png");
  }
}

void draw() {
  // establish if randomization should be executed this frame
  exec_randomize = true;

  // Set the background to random pixels or white:
  if ((rand_background == "pixels") && (second_stage == false)) {
      image(images[frameCount-1], 0, 0, width, height);
    } else if ((rand_background == "whole") && (second_stage == false)) {
      background(color(random(100, 200), random(100, 200), random(100, 200)));
    } else if ((rand_background == "white") && (second_stage == false)) {
      background(255);
    } else if (second_stage == true) {
      background(255); // white anyways
  }

  // Adjust position randomly
    if (shape == "cross") {
    posX = random(1) * width; // Initial X position
    posY = 4 * height / 8; // Initial Y position
  } else if (shape == "rectangle") {
    float pos = random(1);
    posX = pos * width; // Initial X position
    posY = pos * height; // Initial Y position
  } else if (shape == "ellipse") {
    posX = 4 * width / 8; // Initial X position
    posY = random(1) * height; // Initial Y position
  } else {
    println("Error: shape not defined.");
  }

  // Adjust size randomly
  if (rand_size && exec_randomize) {
    // original min = height/3 = 16.7, max = height/1.2 = 41.7
    // new min = 15, max = 45, skip 25 - 35
    // leave out test sizes == 27-33
    d = random(15, 40);
    s_wid_divisor = random(2, 4);
    if ((shape == "cross") || (shape == "rectangle")) {s_wid_divisor = s_wid_divisor * 1.5;} 
    // for debug with window_size != 50
    if (width != 50.0) {d = d * (width / 50.0);}
    s_len = d;
    s_wid = s_len / s_wid_divisor;
  }

  // Adjust color randomly
  if (rand_color && exec_randomize) {
    // original color = random(255) x 3
    // new color = random(255) x 3, if val between 100 and 200, randomize again
    // leave out test colors == random(255) x 3, if val NOT between 120 and 180, randomize again
    a = random(255);
    b = random(255);
    c = random(255);
    e_color = color(a, b, c);
    fill(e_color);
  }

  // Adjust rotation randomly
  if (rand_rotation && exec_randomize) {
    // original theta = random(0, PI/2) aka 8*PI/16
    // new theta = random(0, PI/2), if val between 3*PI/16 and 5*PI/16, randomize again
    // leave out test rotations == random(0, PI), if val NOT between 3*PI/16 and 5*PI/16, randomize again
    th = random(PI/2);
    theta = th;
  }

  // Adjust thickness randomly
  if (rand_thickness && exec_randomize) {
    // original thickness = random(1, 5)
    // new thickness = random(1, 5), if val between 2.5 and 3.5, randomize again
    // leave out test thicknesses == random(1, 5), if val NOT between 2.5 and 3.5, randomize again
    thk = random(1, s_wid/3);
    thickness = thk;
    strokeWeight(thickness);
  }

  pushMatrix(); // Save the current drawing style settings and transformations
  translate(posX, posY); // Move the origin to the new position
  rotate(theta); // Rotate the cross

  // Draw shape
  if (shape == "cross") {
    // Draw the cross lines
    rect(0, 0, s_wid, s_len); // Vertical line
    rect(0, 0, s_len, s_wid); // Horizontal line
    } else if (shape == "rectangle") {
      // Draw the rectangle
      rect(0, 0, s_len, s_wid);
    } else if (shape == "ellipse") {
      // Draw the ellipse
      ellipse(0, 0, s_wid, s_len);
    } else {
      println("Error: shape not defined.");
  }
  
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
    // int n = 30;
    // display_circles(n, n, width/(n*3)); // Display a grid of circles
    for (int i = 0; i < num_occlusions; i++) {
      pushMatrix();
      translate(width/2, height/2);
      rotate(occ_rot[i]);
      if (second_stage == false) {
        occ_colors[i] = color(random(255), random(255), random(255));
      } else {
        occ_colors[i] = color(127, 127, 127); // switch to gray occlusions for second stage
      }
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
  
  else if (save_frames == true) {
    if (!second_stage) {saveFrame("frames/general_" + shape + "_static_1st_stage/###.png");}
    else if (second_stage) {saveFrame("frames/general_" + shape + "_static_2nd_stage/###.png");}
    else {println("Error: stage not defined."); exit();}
    if (frameCount == num_frames) {
      exit();
    }
  }

  else {
    if (frameCount == num_frames) {
      exit();
    }
  }
  
}
