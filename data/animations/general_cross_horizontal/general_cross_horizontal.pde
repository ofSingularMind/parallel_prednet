float window_scale = 1.0;
float posX;
float posY;
float theta = 0; // Rotation angle of the cross
float r_len; // length of the cross
float r_th; // thickness of the cross
float thickness = 3; // stroke thickness of the cross
color e_color = color(0, 0, 0); // Color of the cross
float speed; // Speed of vertical movement
int frame_rate; //<>//
int num_frames;
PImage[] images;
float d; // Random size value
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
// int randomizationRate = 20;
int h = 50;
int w = 50;
boolean rand_size = true;
boolean rand_color = true;
boolean rand_thickness = true;
boolean rand_rotation = true;
boolean rand_occlusions = true;
boolean rand_background = true;

boolean save_gif = false; // only set save_gif or save_frames to true, not both, or both to false
boolean save_frames = false;

boolean train_mode = false; // just flip this one to switch between train and test modes
boolean test_mode = !train_mode;

boolean exec_randomize = true;
boolean flip = false;

import gifAnimation.*;

GifMaker gifExport;

public void settings() {
  size(w, h); // Set the size of the window, w, h
}

void setup() {
  // Set the frame rate and the number of frames to be saved per mode
  if (save_gif && save_frames) {println("Error: save_gif and save_frames cannot both be true."); exit();}
  if (save_gif == true) {num_frames = 150; frame_rate = 1000;}
  else if (save_frames == true) {num_frames = 5000; frame_rate = 1000;}
  else {num_frames = 1000; frame_rate = 10;}
  images = new PImage[num_frames];
  
  r_len = height / 3; // Initial length of the cross
  r_th = r_len / 4; // Initial thickness of the cross
  speed = width / 10; // Initial speed of vertical movement
  posX = width / 2; // Initial X position
  posY = height / 2; // Initial Y position
  fill(e_color); // Set the inital fill color to black
  strokeWeight(thickness);  // Set the initial stroke thickness
  rectMode(CENTER);
  
  frameRate(frame_rate);
  if (save_gif == true) {
    if (train_mode) {gifExport = new GifMaker(this, "general_cross_horizontal_train.gif");}
    else if (test_mode) {gifExport = new GifMaker(this, "general_cross_horizontal_test.gif");}
    else {println("Error: train_mode and test_mode not defined."); exit();}
    gifExport.setRepeat(0); // make it an "endless" animation
    gifExport.setTransparent(255); // make white the transparent color -- match browser bg color
    gifExport.setDelay(1000/frame_rate);  //12fps in ms
  }
  for (int i = 0; i < num_frames; i++) {
    images[i] = loadImage("/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/"+nf(h)+"x"+nf(w)+"/" + nf(i%1000, 3) + ".png");
  }
}

void draw() {
  // establish if randomization should be executed this frame
  if (frameCount == 1) {
    exec_randomize = true;
    } else if (posX < width / 2) {
      exec_randomize = false;
      flip = true;
    } else if ((posX >= width / 2) && (flip == true)){
      exec_randomize = true;
      flip = false;
    }

  // Set the background to random pixels or white:
  if (rand_background) {
    image(images[frameCount-1], 0, 0, width, height);
  } else {
    background(255);
  }

  // Adjust size randomly
  if (rand_size && exec_randomize) {
    // original min = height/3 = 16.7, max = height/1.2 = 41.7
    // new min = 15, max = 45, skip 25 - 35
    // leave out test sizes == 27-33
    d = random(15, 45);
    if (train_mode == true) {
      while (d >= 25 && d <= 35) {d = random(15, 45);}
    } else if (test_mode == true) {
      while (d < 27 || d > 33) {d = random(27, 33);}
    } else {
      println("Error: train_mode and test_mode not defined."); exit();
    }
    // for debug with window_size != 50
    if (width != 50.0) {d = d * (width / 50.0);}
    r_len = d;
    r_th = r_len / 4;
  }

  // Adjust color randomly
  if (rand_color && exec_randomize) {
    // original color = random(255) x 3
    // new color = random(255) x 3, if val between 100 and 200, randomize again
    // leave out test colors == random(255) x 3, if val NOT between 120 and 180, randomize again
    a = random(255);
    b = random(255);
    c = random(255);
    if (train_mode == true) {
      while (a >= 100 && a <= 200) {a = random(255);}
      while (b >= 100 && b <= 200) {b = random(255);}
      while (c >= 100 && c <= 200) {c = random(255);}
    } else if (test_mode == true) {
      while (a < 120 || a > 180) {a = random(120, 180);}
      while (b < 120 || b > 180) {b = random(120, 180);}
      while (c < 120 || c > 180) {c = random(120, 180);}
    } else {
      println("Error: train_mode and test_mode not defined."); exit();
    }
    e_color = color(a, b, c);
    fill(e_color);
  }

  // Adjust rotation randomly
  if (rand_rotation && exec_randomize) {
    // original theta = random(0, PI/2) aka 8*PI/16
    // new theta = random(0, PI/2), if val between 3*PI/16 and 5*PI/16, randomize again
    // leave out test rotations == random(0, PI), if val NOT between 3*PI/16 and 5*PI/16, randomize again
    th = random(PI/2);
    if (train_mode == true) {
      while (th >= 3*PI/16 && th <= 5*PI/16) {th = random(PI/2);}
    } else if (test_mode == true) {
      while (th < 3*PI/16 || th > 5*PI/16) {th = random(3*PI/16, 5*PI/16);}
    } else {
      println("Error: train_mode and test_mode not defined."); exit();
    }
    theta = th;
  }

  // Adjust thickness randomly
  if (rand_thickness && exec_randomize) {
    // original thickness = random(1, 5)
    // new thickness = random(1, 5), if val between 2.5 and 3.5, randomize again
    // leave out test thicknesses == random(1, 5), if val NOT between 2.5 and 3.5, randomize again
    thk = random(1, 5);
    if (train_mode == true) {
      while (thk >= 2.5 && thk <= 3.5) {thk = random(1, 5);}
    } else if (test_mode == true) {
      while (thk < 2.5 || thk > 3.5) {thk = random(2.5, 3.5);}
    } else {
      println("Error: train_mode and test_mode not defined."); exit();
    }
    thickness = thk;
    strokeWeight(thickness);
  }

  if (exec_randomize) {
    // reset randomizer so it is only used once per cycle
    exec_randomize = false;
  }

  // Update position
  posX += speed;
  if (posX - r_len / 2 > width) { // Reset position when it goes beyond the screen
    posX = -r_len / 2;
  }

  pushMatrix(); // Save the current drawing style settings and transformations
  translate(posX, posY); // Move the origin to the new position
  rotate(theta); // Rotate the cross

  // Draw the cross lines
  rect(0, 0, r_th, r_len); // Vertical line
  rect(0, 0, r_len, r_th); // Horizontal line
  
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
    if (train_mode) {saveFrame("frames/general_cross_horizontal/###.png");}
    else if (test_mode) {saveFrame("frames/general_cross_horizontal_test/###.png");}
    else {println("Error: train_mode and test_mode not defined."); exit();}
    if (frameCount == num_frames) {
      exit();
    }
  }
}
